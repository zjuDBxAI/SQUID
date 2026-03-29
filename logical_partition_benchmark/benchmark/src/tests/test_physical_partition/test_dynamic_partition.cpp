#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pqxx/pqxx>

#include "benchmark_utils.h"
#include "independent_hnsw_index.h"

using pointer_benchmark::IndependentHNSWIndex;
using pointer_benchmark::SharedVectorTable;

namespace {
std::string make_roles_key(std::vector<int> roles) {
    std::sort(roles.begin(), roles.end());
    std::string key;
    for (size_t i = 0; i < roles.size(); ++i) {
        if (i > 0) key.push_back(',');
        key += std::to_string(roles[i]);
    }
    return key;
}

int parse_ef_search(int argc, char** argv, int default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--ef-search=", 0) == 0) {
            std::string value = arg.substr(std::string("--ef-search=").size());
            try {
                return std::max(1, std::stoi(value));
            } catch (...) {
                continue;
            }
        }
        if (arg == "--ef-search" || arg == "-e") {
            if (i + 1 < argc) {
                std::string value(argv[++i]);
                try {
                    return std::max(1, std::stoi(value));
                } catch (...) {
                    continue;
                }
            }
        }
    }
    return default_value;
}

int parse_warmup_iterations(int argc, char** argv, int default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--warmup=", 0) == 0) {
            std::string value = arg.substr(std::string("--warmup=").size());
            try {
                return std::max(0, std::stoi(value));
            } catch (...) {
                continue;
            }
        }
        if (arg == "--warmup" || arg == "-w") {
            if (i + 1 < argc) {
                std::string value(argv[++i]);
                try {
                    return std::max(0, std::stoi(value));
                } catch (...) {
                    continue;
                }
            }
        }
    }
    return default_value;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Test: Independent HNSW - Dynamic Partition (Physical)" << std::endl;
        std::cout << "========================================\n" << std::endl;

        const auto root = benchmark_utils::project_root();
        const auto conn_info = benchmark_utils::load_connection_string(root / "config.json");
        auto queries = benchmark_utils::load_queries(root / "basic_benchmark/query_dataset.json");
        if (queries.empty()) {
            throw std::runtime_error("Query dataset is empty");
        }
        std::cout << "Loaded " << queries.size() << " queries" << std::endl;

        const int dimension = static_cast<int>(queries.front().query_vector.size());

        auto hnsw_config = benchmark_utils::load_hnsw_config();
        const int M = hnsw_config.M;
        const int efConstruction = hnsw_config.ef_construction;
        std::cout << "HNSW config: M=" << M << ", ef_construction=" << efConstruction << std::endl;

        const int efSearch = parse_ef_search(argc, argv, 200);
        const int warmup_iterations = parse_warmup_iterations(argc, argv, 0);
        std::cout << "ef_search: " << efSearch << std::endl;
        std::cout << "warmup iterations: " << warmup_iterations << std::endl;

        const std::filesystem::path base_dir = benchmark_utils::load_index_storage_path();
        std::cout << "Index storage path: " << base_dir << std::endl;
        const auto dynamic_dir = base_dir / "physical_dynamic_partition";
        std::filesystem::create_directories(dynamic_dir);

        // Load shared table for ground truth computation only
        const auto shared_cache = dynamic_dir / "shared_vectors.bin";
        auto shared_table = benchmark_utils::load_shared_table(dimension, shared_cache, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        const auto partitions = benchmark_utils::fetch_tables_with_prefix(conn_info, "documentblocks_partition_");
        std::cout << "Found " << partitions.size() << " dynamic partitions" << std::endl;

        std::unordered_map<std::string, std::filesystem::path> partition_dirs;
        std::vector<std::pair<std::string, benchmark_utils::HNSWGraphStats>> graph_breakdown;
        size_t total_vector_bytes = 0;

        for (const auto& table : partitions) {
            auto partition_dir = dynamic_dir / table;
            std::filesystem::create_directories(partition_dir);

            IndependentHNSWIndex index(dimension, M, efConstruction);
            if (!index.load_components(partition_dir.string())) {
                std::cout << "  Building " << table << "..." << std::endl;
                index.build_from_partition(conn_info, table);
                index.save_components(partition_dir.string());
            } else {
                std::cout << "  Loaded " << table << " from disk" << std::endl;
            }

            index.set_ef_search(efSearch);
            size_t mem_bytes = index.memory_bytes();
            total_vector_bytes += mem_bytes;

            benchmark_utils::HNSWGraphStats stats{};
            stats.nodes = index.size();
            graph_breakdown.emplace_back(table, stats);

            std::cout << "    " << index.size() << " vectors, memory="
                      << (mem_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

            partition_dirs.emplace(table, partition_dir);
        }

        const size_t total_bytes = total_vector_bytes;

        std::cout << "\n--- Storage ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total vectors+graphs: " << (total_vector_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Total storage: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

        auto user_roles = benchmark_utils::load_user_roles(conn_info);
        auto doc_roles = benchmark_utils::load_document_roles(conn_info);
        bool ground_truth_from_cache = false;
        auto ground_truth = benchmark_utils::load_or_compute_ground_truth(
            queries, shared_table, user_roles, doc_roles, &ground_truth_from_cache);
        if (ground_truth_from_cache) {
            std::cout << "Loaded ground truth from cache" << std::endl;
        } else {
            std::cout << "Computed ground truth and saved to cache" << std::endl;
        }

        std::unordered_map<std::string, std::vector<std::string>> combo_to_partitions;
        {
            pqxx::connection conn(conn_info);
            pqxx::work txn(conn);
            auto res = txn.exec("SELECT comb_role, partition_id FROM CombRolePartitions");
            for (const auto& row : res) {
                auto comb = benchmark_utils::parse_pg_int_array(row[0].as<std::string>());
                if (comb.empty()) continue;
                int partition_id = row[1].as<int>();
                std::string partition_name = "documentblocks_partition_" + std::to_string(partition_id);

                std::string key = make_roles_key(comb);
                combo_to_partitions[key].push_back(partition_name);
            }
            txn.commit();
        }

        std::vector<double> query_times;
        std::vector<double> recalls;
        query_times.reserve(queries.size());
        recalls.reserve(queries.size());

        std::cout << "\n--- Query Performance ---" << std::endl;
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto& query = queries[i];

            std::vector<int> query_roles;
            {
                pqxx::connection conn(conn_info);
                pqxx::work txn(conn);
                auto res = txn.exec("SELECT role_id FROM UserRoles WHERE user_id = " + txn.quote(query.user_id));
                txn.commit();
                query_roles.reserve(res.size());
                for (const auto& row : res) {
                    int role_id = row[0].as<int>();
                    if (std::find(query_roles.begin(), query_roles.end(), role_id) == query_roles.end()) {
                        query_roles.push_back(role_id);
                    }
                }
                if (query_roles.empty()) {
                    auto cache_it = user_roles.find(query.user_id);
                    if (cache_it != user_roles.end()) {
                        query_roles.assign(cache_it->second.begin(), cache_it->second.end());
                    }
                }
            }
            std::unordered_set<int> query_role_set(query_roles.begin(), query_roles.end());

            std::vector<std::pair<std::pair<int, int>, double>> candidates;
            std::vector<std::pair<int, int>> results;
            double time_ms = 0.0;

            if (!query_roles.empty()) {
                auto role_key = query_roles;
                const auto key = make_roles_key(role_key);
                auto combo_it = combo_to_partitions.find(key);

                if (combo_it != combo_to_partitions.end()) {
                    const float* query_data = query.query_vector.data();

                    for (const auto& partition_name : combo_it->second) {
                        auto dir_it = partition_dirs.find(partition_name);
                        if (dir_it == partition_dirs.end()) continue;

                        IndependentHNSWIndex index(dimension, M, efConstruction);
                        if (!index.load_components(dir_it->second.string())) {
                            std::cerr << "  Warning: failed to load " << partition_name
                                      << ", rebuilding..." << std::endl;
                            index.build_from_partition(conn_info, partition_name);
                            index.save_components(dir_it->second.string());
                        }
                        index.set_ef_search(efSearch);

                        const int partition_topk = std::min(
                            std::max(query.topk, efSearch),
                            static_cast<int>(index.size()));

                        for (int warm = 0; warm < warmup_iterations; ++warm) {
                            index.search(query.query_vector.data(), partition_topk);
                        }
                        auto search_start = std::chrono::high_resolution_clock::now();
                        auto hits = index.search(query.query_vector.data(), partition_topk);
                        auto search_end = std::chrono::high_resolution_clock::now();
                        time_ms += std::chrono::duration<double, std::milli>(search_end - search_start).count();

                        for (const auto& doc_block : hits) {
                            auto doc_role_it = doc_roles.find(doc_block.first);
                            if (doc_role_it == doc_roles.end()) {
                                continue;
                            }
                            bool allowed = false;
                            for (int role : doc_role_it->second) {
                                if (query_role_set.count(role) > 0) {
                                    allowed = true;
                                    break;
                                }
                            }
                            if (!allowed) {
                                continue;
                            }

                            const float* vec = shared_table->get_vector(doc_block.first, doc_block.second);
                            if (!vec) {
                                continue;
                            }

                            double dist = 0.0;
                            for (int d = 0; d < dimension; ++d) {
                                double diff = static_cast<double>(query_data[d]) - static_cast<double>(vec[d]);
                                dist += diff * diff;
                            }
                            candidates.emplace_back(doc_block, dist);
                        }
                    }

                    std::sort(candidates.begin(), candidates.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });

                    results.reserve(static_cast<size_t>(query.topk));
                    std::unordered_set<SharedVectorTable::DocBlockId, SharedVectorTable::DocBlockIdHash> seen;
                    for (const auto& [doc_block, _] : candidates) {
                        if (seen.insert(doc_block).second) {
                            results.push_back(doc_block);
                            if (results.size() == static_cast<size_t>(query.topk)) {
                                break;
                            }
                        }
                    }

                }
            }

            query_times.push_back(time_ms);
            recalls.push_back(benchmark_utils::calculate_recall(results, ground_truth[i]));

            if ((i + 1) % 100 == 0) {
                std::cout << "  Processed " << (i + 1) << "/" << queries.size() << std::endl;
            }
        }

        const double avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                                static_cast<double>(query_times.size());
        const double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                  static_cast<double>(recalls.size());

        auto times_copy = query_times;
        const double p50 = benchmark_utils::percentile(times_copy, 0.5);
        const double p90 = benchmark_utils::percentile(times_copy, 0.9);
        const double p95 = benchmark_utils::percentile(times_copy, 0.95);
        const double p99 = benchmark_utils::percentile(times_copy, 0.99);

        std::cout << std::setprecision(2);
        std::cout << "Avg time: " << avg_time << " ms" << std::endl;
        std::cout << "P50: " << p50 << " ms, P90: " << p90 << " ms, P95: " << p95
                  << " ms, P99: " << p99 << " ms" << std::endl;
        std::cout << "Avg recall: " << avg_recall << std::endl;

        benchmark_utils::json graph_breakdown_json = benchmark_utils::json::array();
        for (const auto& [name, stats] : graph_breakdown) {
            graph_breakdown_json.push_back({
                {"partition", name},
                {"nodes", stats.nodes}
            });
        }

        benchmark_utils::json output = {
            {"strategy", "physical_dynamic_partition"},
            {"storage", {
                {"vector_bytes", total_vector_bytes},
                {"graph_bytes", 0},
                {"total_bytes", total_bytes},
                {"vector_mb", total_vector_bytes / 1024.0 / 1024.0},
                {"graph_mb", 0.0},
                {"total_mb", total_bytes / 1024.0 / 1024.0},
                {"graph_breakdown", graph_breakdown_json}
            }},
            {"performance", {
                {"num_queries", queries.size()},
                {"avg_time_ms", avg_time},
                {"p50_ms", p50},
                {"p90_ms", p90},
                {"p95_ms", p95},
                {"p99_ms", p99},
                {"avg_recall", avg_recall}
            }},
            {"config", {
                {"M", M},
                {"efConstruction", efConstruction},
                {"efSearch", efSearch},
                {"num_partitions", partitions.size()}
            }}
        };

        const auto output_path = root / "logical_partition_benchmark/benchmark/src/physical_dynamic_partition_results.json";
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream out(output_path);
        out << output.dump(2);
        out.close();

        std::cout << "\n✓ Results saved to " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
