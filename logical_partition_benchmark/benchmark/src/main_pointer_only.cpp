#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

#include "shared_vector_table.h"
#include "pointer_hnsw_index.h"
#include "benchmark_utils.h"
#include <faiss/IndexHNSW.h>
#include <pqxx/pqxx>

using namespace pointer_benchmark;
using json = nlohmann::json;
using benchmark_utils::HNSWGraphStats;
using benchmark_utils::Query;
using benchmark_utils::calculate_recall;
using benchmark_utils::compute_ground_truth;
using benchmark_utils::compute_hnsw_graph_stats;
using benchmark_utils::fetch_tables_with_prefix;
using benchmark_utils::load_document_roles;
using benchmark_utils::load_queries;
using benchmark_utils::load_shared_table;
using benchmark_utils::load_user_roles;
using benchmark_utils::percentile;

struct Config {
    std::string db_host = "localhost";
    std::string db_port = "5432";
    std::string db_name = "your_db";
    std::string db_user = "your_user";
    std::string db_password = "your_password";

    int M = 32;
    int efConstruction = 200;
    int efSearch = 50;
    int topk = 10;
    int warmup_iterations = 10;
    int test_iterations = 3;
    bool pointer_copy_baseline = false;
    bool dump_index_sizes = false;
    std::string save_index_dir;
};

Config load_config(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file at " + config_path);
    }
    json j;
    file >> j;

    Config cfg;
    if (j.contains("host")) cfg.db_host = j["host"];
    if (j.contains("port")) {
        if (j["port"].is_string()) {
            cfg.db_port = j["port"].get<std::string>();
        } else {
            cfg.db_port = std::to_string(j["port"].get<int>());
        }
    }
    if (j.contains("dbname")) cfg.db_name = j["dbname"];
    else if (j.contains("database")) cfg.db_name = j["database"];
    if (j.contains("user")) cfg.db_user = j["user"];
    if (j.contains("password")) cfg.db_password = j["password"];
    if (j.contains("pointer_copy_baseline")) {
        cfg.pointer_copy_baseline = j["pointer_copy_baseline"].get<bool>();
    }
    if (j.contains("dump_index_sizes")) {
        cfg.dump_index_sizes = j["dump_index_sizes"].get<bool>();
    }
    if (j.contains("save_index_dir")) {
        cfg.save_index_dir = j["save_index_dir"].get<std::string>();
    }
    return cfg;
}

std::string get_connection_string(const Config& cfg) {
    return "host=" + cfg.db_host +
           " port=" + cfg.db_port +
           " dbname=" + cfg.db_name +
           " user=" + cfg.db_user +
           " password=" + cfg.db_password;
}

int main() {
    try {
        std::filesystem::path project_root = std::filesystem::current_path().parent_path().parent_path().parent_path();

        auto resolve_from_candidates = [&](const std::vector<std::string>& rel_paths) -> std::filesystem::path {
            for (const auto& rel : rel_paths) {
                auto candidate = project_root / rel;
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            std::string error = "Failed to locate file. Checked:";
            for (const auto& rel : rel_paths) {
                error += " " + (project_root / rel).string();
            }
            throw std::runtime_error(error);
        };

        Config cfg = load_config((project_root / "config.json").string());

        auto benchmark_index_path = benchmark_utils::load_index_storage_path();
        if (cfg.save_index_dir.empty()) {
            cfg.save_index_dir = benchmark_index_path.string();
        }

        std::cout << "Index storage path: " << cfg.save_index_dir << std::endl;
        std::string conn_info = get_connection_string(cfg);

        std::filesystem::path query_path = project_root / "basic_benchmark/query_dataset.json";
        if (!std::filesystem::exists(query_path)) {
            throw std::runtime_error(
                "Required query dataset missing: " + query_path.string() +
                ". Please restore or regenerate basic_benchmark/query_dataset.json."
            );
        }
        auto queries = load_queries(query_path);
        if (queries.empty()) {
            throw std::runtime_error("No queries loaded from " + query_path.string());
        }
        std::cout << "Loaded " << queries.size() << " queries from " << query_path << std::endl;

        std::cout << "\n=== DEBUG: Query File Info ===" << std::endl;
        std::cout << "First query user_id: " << queries[0].user_id << std::endl;
        std::cout << "First query topk: " << queries[0].topk << std::endl;
        std::cout << "First query vector (first 5): ";
        for (int i = 0; i < 5; i++) {
            std::cout << queries[0].query_vector[i] << " ";
        }
        std::cout << "\n========================\n" << std::endl;

        int dimension = queries[0].query_vector.size();

        std::filesystem::path base_dir(cfg.save_index_dir);
        std::filesystem::create_directories(base_dir);
        std::filesystem::path pointer_shared_dir = base_dir / "pointer_shared";
        std::filesystem::path pointer_copy_dir = base_dir / "pointer_copy";
        std::filesystem::create_directories(pointer_shared_dir);
        std::filesystem::create_directories(pointer_copy_dir);

        std::filesystem::path shared_vectors_path = pointer_shared_dir / "shared_vectors.bin";
        auto shared_table = load_shared_table(dimension, shared_vectors_path, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        std::vector<std::string> partitions = fetch_tables_with_prefix(conn_info, "documentblocks_role_");
        std::cout << "Found " << partitions.size() << " role partitions" << std::endl;

        auto user_roles = load_user_roles(conn_info);
        auto doc_roles = load_document_roles(conn_info);

        std::string gt_cache_file = (project_root / "pointer_benchmark/ground_truth_cache.json").string();
        std::vector<std::vector<std::pair<int,int>>> ground_truth;

        auto save_ground_truth = [&](const auto& gt) {
            json gt_json = json::array();
            for (const auto& gt_query : gt) {
                json gt_query_json = json::array();
                for (const auto& [doc_id, block_id] : gt_query) {
                    gt_query_json.push_back({doc_id, block_id});
                }
                gt_json.push_back(gt_query_json);
            }

            std::ofstream out(gt_cache_file);
            out << gt_json.dump(2);
        };

        std::ifstream gt_file(gt_cache_file);
        if (gt_file.good()) {
            std::cout << "\nLoading ground truth from cache..." << std::endl;
            json gt_json;
            gt_file >> gt_json;
            gt_file.close();

            for (const auto& gt_query : gt_json) {
                std::vector<std::pair<int,int>> gt_results;
                for (const auto& result : gt_query) {
                    gt_results.push_back({result[0].get<int>(), result[1].get<int>()});
                }
                ground_truth.push_back(gt_results);
            }
            std::cout << "Loaded ground truth for " << ground_truth.size() << " queries" << std::endl;
        }

        if (ground_truth.size() != queries.size()) {
            if (!ground_truth.empty()) {
                std::cout << "Ground truth cache size mismatch (" << ground_truth.size()
                          << " vs " << queries.size() << "). Recomputing..." << std::endl;
            } else {
                std::cout << "\nComputing ground truth (this will take a while)..." << std::endl;
            }

            ground_truth = compute_ground_truth(queries, shared_table, user_roles, doc_roles);

            std::cout << "Saving ground truth to cache..." << std::endl;
            save_ground_truth(ground_truth);
            std::cout << "Ground truth cached to " << gt_cache_file << std::endl;
        }

        auto ensure_pointer_graphs = [&](bool force_copy,
                                         const std::filesystem::path& dir,
                                         const std::shared_ptr<SharedVectorTable>& shared_table_local) {
            std::filesystem::create_directories(dir);
            for (const auto& partition : partitions) {
                std::filesystem::path graph_path = dir / (partition + "_graph.bin");
                if (std::filesystem::exists(graph_path)) {
                    continue;
                }
                std::cout << "Building pointer graph for " << partition << "..." << std::endl;
                PointerHNSWIndex idx(shared_table_local, cfg.M, cfg.efConstruction);
                idx.set_force_copy_mode(force_copy);
                idx.build_from_partition(conn_info, partition);
                idx.set_ef_search(cfg.efSearch);
                if (!cfg.save_index_dir.empty()) {
                    idx.save_graph(graph_path.string());
                }
            }
        };

        auto evaluate_pointer_runtime = [&](const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                            const std::filesystem::path& pointer_dir,
                                            std::vector<double>& query_times_out,
                                            const std::vector<std::vector<std::pair<int,int>>>& ground_truth_local) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "SCENARIO: Pointer HNSW (Shared Storage)" << std::endl;
            std::cout << "========================================" << std::endl;
            std::vector<double> query_times;
            std::vector<double> recalls;

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                pqxx::connection c(conn_info);
                pqxx::work t(c);
                auto res = t.exec("SELECT role_id FROM UserRoles WHERE user_id = " + t.quote(q.user_id));
                t.commit();

                std::vector<std::pair<int,int>> all_results;
                double total_partition_time = 0.0;

                for (const auto& row : res) {
                    int role_id = row[0].as<int>();
                    std::string part_name = "documentblocks_role_" + std::to_string(role_id);
                    std::filesystem::path graph_path = pointer_dir / (part_name + "_graph.bin");

                    PointerHNSWIndex idx(shared_table_local, cfg.M, cfg.efConstruction);
                    if (!idx.load_graph(graph_path.string())) {
                        std::cerr << "Warning: failed to load pointer graph " << graph_path
                                  << ", rebuilding..." << std::endl;
                        idx.build_from_partition(conn_info, part_name);
                        idx.set_ef_search(cfg.efSearch);
                        if (!cfg.save_index_dir.empty()) {
                            idx.save_graph(graph_path.string());
                        }
                    }
                    idx.set_ef_search(cfg.efSearch);

                    idx.search(q.query_vector.data(), q.topk); // warm-up

                    auto start = std::chrono::high_resolution_clock::now();
                    auto results = idx.search(q.query_vector.data(), q.topk);
                    auto end = std::chrono::high_resolution_clock::now();

                    total_partition_time += std::chrono::duration<double, std::milli>(end - start).count();
                    all_results.insert(all_results.end(), results.begin(), results.end());
                }

                query_times.push_back(total_partition_time);
                const auto& gt = (i < ground_truth_local.size()) ? ground_truth_local[i]
                                                                 : std::vector<std::pair<int,int>>{};
                recalls.push_back(calculate_recall(all_results, gt));
            }

            auto times_copy = query_times;
            query_times_out = query_times;

            double avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                              query_times.size();
            double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                recalls.size();
            double p50 = percentile(times_copy, 0.5);
            double p90 = percentile(times_copy, 0.9);
            double p95 = percentile(times_copy, 0.95);
            double p99 = percentile(times_copy, 0.99);

            std::cout << "Results:" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Avg time: " << avg_time << " ms" << std::endl;
            std::cout << "  P50: " << p50 << " ms, P90: " << p90
                      << " ms, P95: " << p95 << " ms, P99: " << p99 << " ms" << std::endl;
            std::cout << "  Avg recall: " << avg_recall << std::endl;
        };

        auto evaluate_pointer_copy_runtime = [&](const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                                 const std::filesystem::path& pointer_dir,
                                                 const std::vector<std::vector<std::pair<int,int>>>& ground_truth_local) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "SCENARIO: Pointer HNSW (Legacy Copy Mode)" << std::endl;
            std::cout << "========================================" << std::endl;
            std::vector<double> query_times;
            std::vector<double> recalls;

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                pqxx::connection c(conn_info);
                pqxx::work t(c);
                auto res = t.exec("SELECT role_id FROM UserRoles WHERE user_id = " + t.quote(q.user_id));
                t.commit();

                std::vector<std::pair<int,int>> all_results;
                double total_partition_time = 0.0;

                for (const auto& row : res) {
                    int role_id = row[0].as<int>();
                    std::string part_name = "documentblocks_role_" + std::to_string(role_id);
                    std::filesystem::path graph_path = pointer_dir / (part_name + "_graph.bin");

                    PointerHNSWIndex idx(shared_table_local, cfg.M, cfg.efConstruction);
                    idx.set_force_copy_mode(true);
                    if (!idx.load_graph(graph_path.string())) {
                        std::cerr << "Warning: failed to load pointer graph " << graph_path
                                  << ", rebuilding..." << std::endl;
                        idx.build_from_partition(conn_info, part_name);
                        idx.set_ef_search(cfg.efSearch);
                        if (!cfg.save_index_dir.empty()) {
                            idx.save_graph(graph_path.string());
                        }
                    }
                    idx.set_ef_search(cfg.efSearch);

                    idx.search(q.query_vector.data(), q.topk); // warm-up

                    auto start = std::chrono::high_resolution_clock::now();
                    auto results = idx.search(q.query_vector.data(), q.topk);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_partition_time += std::chrono::duration<double, std::milli>(end - start).count();
                    all_results.insert(all_results.end(), results.begin(), results.end());
                }

                query_times.push_back(total_partition_time);
                const auto& gt = (i < ground_truth_local.size()) ? ground_truth_local[i]
                                                                 : std::vector<std::pair<int,int>>{};
                recalls.push_back(calculate_recall(all_results, gt));
            }

            auto times_copy = query_times;
            double avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                              query_times.size();
            double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                recalls.size();
            double p50 = percentile(times_copy, 0.5);
            double p90 = percentile(times_copy, 0.9);
            double p95 = percentile(times_copy, 0.95);
            double p99 = percentile(times_copy, 0.99);

            std::cout << "Results:" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Avg time: " << avg_time << " ms" << std::endl;
            std::cout << "  P50: " << p50 << " ms, P90: " << p90
                      << " ms, P95: " << p95 << " ms, P99: " << p99 << " ms" << std::endl;
            std::cout << "  Avg recall: " << avg_recall << std::endl;
        };

        ensure_pointer_graphs(false, pointer_shared_dir, shared_table);
        if (cfg.pointer_copy_baseline) {
            ensure_pointer_graphs(true, pointer_copy_dir, shared_table);
        }

        std::vector<double> pointer_query_times;
        evaluate_pointer_runtime(shared_table,
                                 pointer_shared_dir,
                                 pointer_query_times,
                                 ground_truth);

        if (cfg.pointer_copy_baseline) {
            evaluate_pointer_copy_runtime(shared_table,
                                          pointer_copy_dir,
                                          ground_truth);
        }

        std::cout << "\nPointer-only benchmark complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
