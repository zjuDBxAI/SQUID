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
#include "independent_hnsw_index.h"
#include "global_hnsw_index.h"
#include "benchmark_utils.h"
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <pqxx/pqxx>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>

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

static json graph_breakdown_to_json(const std::vector<std::pair<std::string, HNSWGraphStats>>& breakdown) {
    json arr = json::array();
    for (const auto& [name, stats] : breakdown) {
        arr.push_back({
                {"partition", name},
                {"nodes", stats.nodes},
                {"neighbor_bytes", stats.neighbor_bytes},
                {"offset_bytes", stats.offsets_bytes},
                {"levels_bytes", stats.levels_bytes},
                {"assign_probas_bytes", stats.assign_probas_bytes},
                {"cum_nneighbor_bytes", stats.cum_nneighbor_bytes},
                {"total_bytes", stats.total_bytes()}});
    }
    return arr;
}

// Configuration
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
    int global_ef_search = 800;
};

// Load config from JSON
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
    if (j.contains("global_ef_search")) {
        cfg.global_ef_search = j["global_ef_search"].get<int>();
    }

    return cfg;
}

// Get connection string
std::string get_connection_string(const Config& cfg) {
    return "host=" + cfg.db_host +
           " port=" + cfg.db_port +
           " dbname=" + cfg.db_name +
           " user=" + cfg.db_user +
           " password=" + cfg.db_password;
}

int main() {
    try {
        // Get project root (build2 -> pointer_benchmark -> HoneyBee-VectorAccess)
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

        // Load configuration
        Config cfg = load_config((project_root / "config.json").string());

        // Use index storage path from benchmark config if available
        auto benchmark_index_path = benchmark_utils::load_index_storage_path();
        if (cfg.save_index_dir.empty()) {
            cfg.save_index_dir = benchmark_index_path.string();
        }
        std::cout << "Index storage path: " << cfg.save_index_dir << std::endl;
        std::string conn_info = get_connection_string(cfg);

        // Load queries (must use basic benchmark dataset)
        std::filesystem::path query_path = project_root / "basic_benchmark/query_dataset.json";
        if (!std::filesystem::exists(query_path)) {
            throw std::runtime_error(
                "Required query dataset missing: " + query_path.string() +
                ". Please restore or regenerate basic_benchmark/query_dataset.json."
            );
        }
        auto queries = load_queries(query_path);
        std::cout << "Loaded " << queries.size() << " queries from " << query_path << std::endl;

        // Debug: print first query info
        std::cout << "\n=== DEBUG: Query File Info ===" << std::endl;
        std::cout << "First query user_id: " << queries[0].user_id << std::endl;
        std::cout << "First query topk: " << queries[0].topk << std::endl;
        std::cout << "First query vector (first 5): ";
        for (int i = 0; i < 5; i++) {
            std::cout << queries[0].query_vector[i] << " ";
        }
        std::cout << "\n========================\n" << std::endl;

        // Get dimension from first query
        int dimension = queries[0].query_vector.size();

        std::filesystem::path base_dir(cfg.save_index_dir);
        std::filesystem::create_directories(base_dir);
        std::filesystem::path pointer_shared_dir = base_dir / "pointer_shared";
        std::filesystem::path pointer_copy_dir = base_dir / "pointer_copy";
        std::filesystem::path global_dir = base_dir / "global";
        std::filesystem::path independent_dir = base_dir / "independent";
        std::filesystem::create_directories(pointer_shared_dir);
        std::filesystem::create_directories(pointer_copy_dir);
        std::filesystem::create_directories(global_dir);
        std::filesystem::create_directories(independent_dir);

        // Load shared vectors from disk if available, otherwise build once
        std::filesystem::path shared_vectors_path = pointer_shared_dir / "shared_vectors.bin";
        auto shared_table = load_shared_table(dimension, shared_vectors_path, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        // Get partition tables
        std::vector<std::string> partitions = fetch_tables_with_prefix(conn_info, "documentblocks_role_");
        std::cout << "Found " << partitions.size() << " role partitions" << std::endl;

        auto user_roles = load_user_roles(conn_info);
        auto doc_roles = load_document_roles(conn_info);

        // Load or compute ground truth
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

        size_t missing_ground_truth = 0;
        size_t total_ground_truth = 0;
        for (const auto& gt_results : ground_truth) {
            total_ground_truth += gt_results.size();
            for (const auto& doc_block : gt_results) {
                if (shared_table->get_internal_index(doc_block.first, doc_block.second) < 0) {
                    missing_ground_truth++;
                }
            }
        }

        if (missing_ground_truth > 0) {
            std::cout << "Warning: " << missing_ground_truth << "/" << total_ground_truth
                      << " ground truth entries missing from shared vector table" << std::endl;
        }
        std::cout << "Loaded ground truth for " << ground_truth.size() << " queries" << std::endl;

        struct PointerMetrics {
            size_t vector_bytes = 0;
            size_t graph_bytes = 0;
            double avg_time = 0.0;
            double avg_recall = 0.0;
            double p50 = 0.0;
            double p90 = 0.0;
            double p95 = 0.0;
            double p99 = 0.0;
            std::vector<std::pair<std::string, HNSWGraphStats>> graph_breakdown;

            size_t total_bytes() const { return vector_bytes + graph_bytes; }
        };

        struct GlobalMetrics {
            size_t vector_bytes = 0;
            size_t graph_bytes = 0;
            double avg_time = 0.0;
            double avg_recall = 0.0;
            double p50 = 0.0;
            double p90 = 0.0;
            double p95 = 0.0;
            double p99 = 0.0;
            HNSWGraphStats graph_stats;

            size_t total_bytes() const { return vector_bytes + graph_bytes; }
        };

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

        auto ensure_independent_components = [&](const std::filesystem::path& dir) {
            std::filesystem::create_directories(dir);
            for (const auto& partition : partitions) {
                std::filesystem::path partition_dir = dir / partition;
                std::filesystem::create_directories(partition_dir);
                std::filesystem::path index_path = partition_dir / "integrated_index.bin";
                std::filesystem::path meta_path = partition_dir / "metadata.bin";
                if (std::filesystem::exists(index_path) && std::filesystem::exists(meta_path)) {
                    continue;
                }
                std::cout << "Building independent index for " << partition << "..." << std::endl;
                IndependentHNSWIndex idx(dimension, cfg.M, cfg.efConstruction);
                idx.build_from_partition(conn_info, partition);
                idx.set_ef_search(cfg.efSearch);
                if (!cfg.save_index_dir.empty()) {
                    idx.save_components(partition_dir.string());
                }
            }
        };

        auto ensure_global_graph = [&](const std::filesystem::path& dir,
                                       const std::shared_ptr<SharedVectorTable>& shared_table_local) {
            std::filesystem::create_directories(dir);
            std::filesystem::path graph_path = dir / "global_graph.bin";
            if (std::filesystem::exists(graph_path)) {
                return;
            }
            std::cout << "Building global HNSW graph..." << std::endl;
            GlobalHNSWIndex global_builder(IndexMode::SHARED_POINTER, shared_table_local, cfg.M, cfg.efConstruction);
            global_builder.build();
            if (!cfg.save_index_dir.empty()) {
                global_builder.save_graph(graph_path.string());
            }
        };

        auto collect_pointer_stats = [&](const std::filesystem::path& dir,
                                         const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                         PointerMetrics& metrics,
                                         std::unordered_map<SharedVectorTable::DocBlockId,
                                                            std::vector<int>,
                                                            SharedVectorTable::DocBlockIdHash>* doc_roles_map_ptr) {
            metrics.vector_bytes = shared_table_local->get_index_flat()->ntotal * dimension * sizeof(float);
            metrics.graph_bytes = 0;
            metrics.graph_breakdown.clear();
            for (const auto& partition : partitions) {
                PointerHNSWIndex idx(shared_table_local, cfg.M, cfg.efConstruction);
                std::filesystem::path graph_path = dir / (partition + "_graph.bin");
                if (!idx.load_graph(graph_path.string())) {
                    std::cerr << "Warning: failed to load pointer graph " << graph_path
                              << ", rebuilding..." << std::endl;
                    idx.build_from_partition(conn_info, partition);
                    if (!cfg.save_index_dir.empty()) {
                        idx.save_graph(graph_path.string());
                    }
                }
                const faiss::IndexHNSW* faiss_index = idx.get_hnsw_index();
                auto stats = compute_hnsw_graph_stats(faiss_index->hnsw);
                metrics.graph_bytes += stats.total_bytes();
                metrics.graph_breakdown.emplace_back(partition, stats);

                if (doc_roles_map_ptr) {
                    size_t pos = partition.find_last_of('_');
                    if (pos != std::string::npos) {
                        int role_id = std::stoi(partition.substr(pos + 1));
                        for (const auto& doc : idx.doc_blocks()) {
                            auto& roles = (*doc_roles_map_ptr)[doc];
                            if (std::find(roles.begin(), roles.end(), role_id) == roles.end()) {
                                roles.push_back(role_id);
                            }
                        }
                    }
                }
            }
        };

        auto collect_global_stats = [&](const std::filesystem::path& graph_path,
                                        const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                        GlobalMetrics& metrics) {
            GlobalHNSWIndex idx(IndexMode::SHARED_POINTER, shared_table_local, cfg.M, cfg.efConstruction);
            if (!idx.load_graph(graph_path.string())) {
                idx.build();
                if (!cfg.save_index_dir.empty()) {
                    idx.save_graph(graph_path.string());
                }
            }
            const faiss::IndexHNSW* faiss_index = idx.get_index();
            auto stats = compute_hnsw_graph_stats(faiss_index->hnsw);
            metrics.vector_bytes = shared_table_local->get_index_flat()->ntotal * dimension * sizeof(float);
            metrics.graph_bytes = stats.total_bytes();
            metrics.graph_stats = stats;
        };

        auto collect_independent_stats = [&](const std::filesystem::path& dir,
                                             PointerMetrics& metrics) {
            metrics.vector_bytes = 0;
            metrics.graph_bytes = 0;
            metrics.graph_breakdown.clear();
            for (const auto& partition : partitions) {
                IndependentHNSWIndex idx(dimension, cfg.M, cfg.efConstruction);
                std::filesystem::path partition_dir = dir / partition;
                if (!idx.load_components(partition_dir.string())) {
                    idx.build_from_partition(conn_info, partition);
                    if (!cfg.save_index_dir.empty()) {
                        idx.save_components(partition_dir.string());
                    }
                }
                metrics.vector_bytes += idx.memory_bytes();
                auto stats = benchmark_utils::HNSWGraphStats{};
                stats.nodes = idx.size();
                metrics.graph_breakdown.emplace_back(partition, stats);
            }
        };

        auto evaluate_pointer_runtime = [&](const std::string& title,
                                            const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                            const std::filesystem::path& pointer_dir,
                                            PointerMetrics& metrics_out,
                                            std::vector<double>& query_times_out) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "SCENARIO 1: " << title << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Memory: " << (metrics_out.total_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  - Stored vectors: " << (metrics_out.vector_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  - Graph structures: " << (metrics_out.graph_bytes / 1024.0 / 1024.0)
                      << " MB" << std::endl;
            size_t pointer_total_nodes = 0;
            for (const auto& [name, stats] : metrics_out.graph_breakdown) {
                pointer_total_nodes += stats.nodes;
            }
            if (pointer_total_nodes > 0) {
                double avg_graph_per_node = static_cast<double>(metrics_out.graph_bytes) /
                                             static_cast<double>(pointer_total_nodes);
                std::cout << "    (" << pointer_total_nodes << " nodes across "
                          << metrics_out.graph_breakdown.size()
                          << " partitions, avg " << (avg_graph_per_node / 1024.0)
                          << " KB per node)" << std::endl;
            }
            if (cfg.dump_index_sizes && !metrics_out.graph_breakdown.empty()) {
                std::cout << "  - Graph breakdown:" << std::endl;
                for (const auto& [name, stats] : metrics_out.graph_breakdown) {
                    std::cout << "    * " << name
                              << ": nodes=" << stats.nodes
                              << ", neighbors=" << (stats.neighbor_bytes / 1024.0 / 1024.0) << " MB"
                              << ", offsets=" << (stats.offsets_bytes / 1024.0 / 1024.0) << " MB"
                              << ", levels=" << (stats.levels_bytes / 1024.0 / 1024.0) << " MB"
                              << std::endl;
                }
            }
            std::cout.unsetf(std::ios_base::floatfield);
            std::cout << std::setprecision(6);
            std::vector<double> query_times;
            std::vector<double> recalls;
            std::vector<long long> all_cycles, all_instr, all_cache_ref, all_cache_miss, all_l1d_loads, all_l1d_miss, all_l1i_loads, all_l1i_miss, all_llc_loads, all_llc_miss;
            std::vector<long long> all_stall_fe, all_stall_be, all_br_miss, all_ctx_sw, all_cpu_mig, all_page_fault;

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

                    {  // Scope for immediate index release
                        PointerHNSWIndex idx(shared_table_local, cfg.M, cfg.efConstruction);
                        // std::shared_ptr<SharedVectorTable> shared_table_scope(
                        //         shared_table_local->clone().release());
                        // PointerHNSWIndex idx(shared_table_scope, cfg.M, cfg.efConstruction);
                        if (!idx.load_graph(graph_path.string())) {
                            idx.build_from_partition(conn_info, part_name);
                            idx.set_ef_search(cfg.efSearch);
                            if (!cfg.save_index_dir.empty()) {
                                idx.save_graph(graph_path.string());
                            }
                        }
                        idx.set_ef_search(cfg.efSearch);

                        // Warm up
                        idx.search(q.query_vector.data(), q.topk);

                        // Setup perf counters
                        struct perf_event_attr pe;
                        memset(&pe, 0, sizeof(pe));
                        pe.size = sizeof(pe);
                        pe.disabled = 1;
                        pe.exclude_kernel = 1;
                        pe.exclude_hv = 1;

                        pe.type = PERF_TYPE_HARDWARE; pe.config = PERF_COUNT_HW_CPU_CYCLES;
                        int fd_cycles = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                        int fd_instr = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
                        int fd_cache_ref = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_CACHE_MISSES;
                        int fd_cache_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
                        int fd_stall_fe = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
                        int fd_stall_be = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_HW_BRANCH_MISSES;
                        int fd_br_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);

                        pe.type = PERF_TYPE_SOFTWARE;
                        pe.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
                        int fd_ctx_sw = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_SW_CPU_MIGRATIONS;
                        int fd_cpu_mig = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = PERF_COUNT_SW_PAGE_FAULTS;
                        int fd_page_fault = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);

                        pe.type = PERF_TYPE_HW_CACHE;
                        pe.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                        int fd_l1d_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                        int fd_l1d_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                        int fd_l1i_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = (PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                        int fd_l1i_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                        int fd_llc_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                        pe.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                        int fd_llc_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);

                        // Reset and enable all counters
                        int fds[] = {fd_cycles, fd_instr, fd_cache_ref, fd_cache_miss, fd_l1d_loads, fd_l1d_miss, fd_l1i_loads, fd_l1i_miss, fd_llc_loads, fd_llc_miss, fd_stall_fe, fd_stall_be, fd_br_miss, fd_ctx_sw, fd_cpu_mig, fd_page_fault};
                        for (int fd : fds) { if (fd >= 0) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); } }

                        // Timed search
                        auto start = std::chrono::high_resolution_clock::now();
                        auto results = idx.search(q.query_vector.data(), q.topk);
                        auto end = std::chrono::high_resolution_clock::now();

                        // Disable and read counters
                        for (int fd : fds) { if (fd >= 0) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0); }
                        long long cycles = 0, instr = 0, cache_ref = 0, cache_miss = 0, l1d_loads = 0, l1d_miss = 0, l1i_loads = 0, l1i_miss = 0, llc_loads = 0, llc_miss = 0;
                        long long stall_fe = 0, stall_be = 0, br_miss = 0, ctx_sw = 0, cpu_mig = 0, page_fault = 0;
                        if (fd_cycles >= 0) read(fd_cycles, &cycles, sizeof(cycles));
                        if (fd_instr >= 0) read(fd_instr, &instr, sizeof(instr));
                        if (fd_cache_ref >= 0) read(fd_cache_ref, &cache_ref, sizeof(cache_ref));
                        if (fd_cache_miss >= 0) read(fd_cache_miss, &cache_miss, sizeof(cache_miss));
                        if (fd_l1d_loads >= 0) read(fd_l1d_loads, &l1d_loads, sizeof(l1d_loads));
                        if (fd_l1d_miss >= 0) read(fd_l1d_miss, &l1d_miss, sizeof(l1d_miss));
                        if (fd_l1i_loads >= 0) read(fd_l1i_loads, &l1i_loads, sizeof(l1i_loads));
                        if (fd_l1i_miss >= 0) read(fd_l1i_miss, &l1i_miss, sizeof(l1i_miss));
                        if (fd_llc_loads >= 0) read(fd_llc_loads, &llc_loads, sizeof(llc_loads));
                        if (fd_llc_miss >= 0) read(fd_llc_miss, &llc_miss, sizeof(llc_miss));
                        if (fd_stall_fe >= 0) read(fd_stall_fe, &stall_fe, sizeof(stall_fe));
                        if (fd_stall_be >= 0) read(fd_stall_be, &stall_be, sizeof(stall_be));
                        if (fd_br_miss >= 0) read(fd_br_miss, &br_miss, sizeof(br_miss));
                        if (fd_ctx_sw >= 0) read(fd_ctx_sw, &ctx_sw, sizeof(ctx_sw));
                        if (fd_cpu_mig >= 0) read(fd_cpu_mig, &cpu_mig, sizeof(cpu_mig));
                        if (fd_page_fault >= 0) read(fd_page_fault, &page_fault, sizeof(page_fault));

                        // Collect perf stats
                        all_cycles.push_back(cycles);
                        all_instr.push_back(instr);
                        all_cache_ref.push_back(cache_ref);
                        all_cache_miss.push_back(cache_miss);
                        all_l1d_loads.push_back(l1d_loads);
                        all_l1d_miss.push_back(l1d_miss);
                        all_l1i_loads.push_back(l1i_loads);
                        all_l1i_miss.push_back(l1i_miss);
                        all_llc_loads.push_back(llc_loads);
                        all_llc_miss.push_back(llc_miss);
                        all_stall_fe.push_back(stall_fe);
                        all_stall_be.push_back(stall_be);
                        all_br_miss.push_back(br_miss);
                        all_ctx_sw.push_back(ctx_sw);
                        all_cpu_mig.push_back(cpu_mig);
                        all_page_fault.push_back(page_fault);

                        // Print perf stats for first query
                        if (i == 0) {
                            std::cout << "\n=== Perf Stats (First Query) ===" << std::endl;
                            std::cout << "Cycles: " << cycles << std::endl;
                            std::cout << "Instructions: " << instr << std::endl;
                            std::cout << "IPC: " << (cycles > 0 ? (double)instr/cycles : 0.0) << std::endl;
                            std::cout << "Stalled cycles (frontend): " << stall_fe << " (" << (cycles > 0 ? (double)stall_fe*100.0/cycles : 0.0) << "%)" << std::endl;
                            std::cout << "Stalled cycles (backend): " << stall_be << " (" << (cycles > 0 ? (double)stall_be*100.0/cycles : 0.0) << "%)" << std::endl;
                            std::cout << "Branch misses: " << br_miss << std::endl;
                            std::cout << "Context switches: " << ctx_sw << std::endl;
                            std::cout << "CPU migrations: " << cpu_mig << std::endl;
                            std::cout << "Page faults: " << page_fault << std::endl;
                            std::cout << "Cache refs: " << cache_ref << ", misses: " << cache_miss
                                      << " (" << (cache_ref > 0 ? (double)cache_miss*100.0/cache_ref : 0.0) << "%)" << std::endl;
                            std::cout << "L1D loads: " << l1d_loads << ", misses: " << l1d_miss
                                      << " (" << (l1d_loads > 0 ? (double)l1d_miss*100.0/l1d_loads : 0.0) << "%)" << std::endl;
                            std::cout << "L1I loads: " << l1i_loads << ", misses: " << l1i_miss
                                      << " (" << (l1i_loads > 0 ? (double)l1i_miss*100.0/l1i_loads : 0.0) << "%)" << std::endl;
                            std::cout << "LLC loads: " << llc_loads << ", misses: " << llc_miss
                                      << " (" << (llc_loads > 0 ? (double)llc_miss*100.0/llc_loads : 0.0) << "%)" << std::endl;
                            std::cout << "================================" << std::endl;
                        }

                        // Close all fds
                        for (int fd : fds) { if (fd >= 0) close(fd); }

                        total_partition_time += std::chrono::duration<double, std::milli>(end - start).count();
                        all_results.insert(all_results.end(), results.begin(), results.end());
                    }  // idx released here, graph memory freed immediately
                }

                query_times.push_back(total_partition_time);

                // Debug: print first query's results
                if (i == 0) {
                    std::cout << "\n=== DEBUG: First Query (Pointer HNSW) ===" << std::endl;
                    std::cout << "Total search results: " << all_results.size() << std::endl;
                    std::cout << "Search results (first 3):" << std::endl;
                    for (size_t j = 0; j < std::min(size_t(3), all_results.size()); j++) {
                        std::cout << "  [" << j << "] doc_id=" << all_results[j].first
                                  << ", block_id=" << all_results[j].second << std::endl;
                    }
                    std::cout << "Ground truth size: " << ground_truth[i].size() << std::endl;
                    std::cout << "Ground truth (first 3):" << std::endl;
                    for (size_t j = 0; j < std::min(size_t(3), ground_truth[i].size()); j++) {
                        std::cout << "  [" << j << "] doc_id=" << ground_truth[i][j].first
                                  << ", block_id=" << ground_truth[i][j].second << std::endl;
                    }
                    std::cout << "========================\n" << std::endl;
                }

                recalls.push_back(calculate_recall(all_results, ground_truth[i]));
            }

            metrics_out.avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                                   query_times.size();
            metrics_out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                     recalls.size();
            auto times_copy = query_times;
            metrics_out.p50 = percentile(times_copy, 0.5);
            metrics_out.p90 = percentile(times_copy, 0.9);
            metrics_out.p95 = percentile(times_copy, 0.95);
            metrics_out.p99 = percentile(times_copy, 0.99);

            query_times_out = query_times;

            std::cout << "Results:" << std::endl;
            std::cout << "  Avg time: " << metrics_out.avg_time << " ms" << std::endl;
            std::cout << "  P50: " << metrics_out.p50 << " ms, P90: " << metrics_out.p90
                      << " ms, P95: " << metrics_out.p95 << " ms, P99: " << metrics_out.p99 << " ms"
                      << std::endl;
            std::cout << "  Avg recall: " << metrics_out.avg_recall << std::endl;

            // Print average perf stats
            auto avg = [](const std::vector<long long>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0LL) / (double)v.size(); };
            double avg_cycles = avg(all_cycles), avg_instr = avg(all_instr), avg_cache_ref = avg(all_cache_ref);
            double avg_cache_miss = avg(all_cache_miss), avg_l1d_loads = avg(all_l1d_loads), avg_l1d_miss = avg(all_l1d_miss);
            double avg_l1i_loads = avg(all_l1i_loads), avg_l1i_miss = avg(all_l1i_miss);
            double avg_llc_loads = avg(all_llc_loads), avg_llc_miss = avg(all_llc_miss);
            double avg_stall_fe = avg(all_stall_fe), avg_stall_be = avg(all_stall_be), avg_br_miss = avg(all_br_miss);
            double avg_ctx_sw = avg(all_ctx_sw), avg_cpu_mig = avg(all_cpu_mig), avg_page_fault = avg(all_page_fault);
            std::cout << "\n=== Average Perf Stats (All Queries) ===" << std::endl;
            std::cout << "Avg Cycles: " << avg_cycles << std::endl;
            std::cout << "Avg Instructions: " << avg_instr << std::endl;
            std::cout << "Avg IPC: " << (avg_cycles > 0 ? avg_instr/avg_cycles : 0.0) << std::endl;
            std::cout << "Avg Stalled cycles (frontend): " << avg_stall_fe << " (" << (avg_cycles > 0 ? avg_stall_fe*100.0/avg_cycles : 0.0) << "%)" << std::endl;
            std::cout << "Avg Stalled cycles (backend): " << avg_stall_be << " (" << (avg_cycles > 0 ? avg_stall_be*100.0/avg_cycles : 0.0) << "%)" << std::endl;
            std::cout << "Avg Branch misses: " << avg_br_miss << std::endl;
            std::cout << "Avg Context switches: " << avg_ctx_sw << std::endl;
            std::cout << "Avg CPU migrations: " << avg_cpu_mig << std::endl;
            std::cout << "Avg Page faults: " << avg_page_fault << std::endl;
            std::cout << "Avg Cache refs: " << avg_cache_ref << ", misses: " << avg_cache_miss
                      << " (" << (avg_cache_ref > 0 ? avg_cache_miss*100.0/avg_cache_ref : 0.0) << "%)" << std::endl;
            std::cout << "Avg L1D loads: " << avg_l1d_loads << ", misses: " << avg_l1d_miss
                      << " (" << (avg_l1d_loads > 0 ? avg_l1d_miss*100.0/avg_l1d_loads : 0.0) << "%)" << std::endl;
            std::cout << "Avg L1I loads: " << avg_l1i_loads << ", misses: " << avg_l1i_miss
                      << " (" << (avg_l1i_loads > 0 ? avg_l1i_miss*100.0/avg_l1i_loads : 0.0) << "%)" << std::endl;
            std::cout << "Avg LLC loads: " << avg_llc_loads << ", misses: " << avg_llc_miss
                      << " (" << (avg_llc_loads > 0 ? avg_llc_miss*100.0/avg_llc_loads : 0.0) << "%)" << std::endl;
            std::cout << "========================================" << std::endl;
        };

        auto evaluate_independent_runtime = [&](const std::string& title,
                                                const std::filesystem::path& independent_dir,
                                                PointerMetrics& metrics_out,
                                                std::vector<double>& query_times_out) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "SCENARIO 3: " << title << std::endl;
            std::cout << "========================================" << std::endl;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Memory: " << (metrics_out.total_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  - Stored vectors: " << (metrics_out.vector_bytes / 1024.0 / 1024.0)
                      << " MB" << std::endl;
            std::cout << "  - Graph structures: " << (metrics_out.graph_bytes / 1024.0 / 1024.0)
                      << " MB" << std::endl;
            size_t total_nodes = 0;
            for (const auto& entry : metrics_out.graph_breakdown) {
                total_nodes += entry.second.nodes;
            }
            if (total_nodes > 0) {
                double avg_graph_per_node = static_cast<double>(metrics_out.graph_bytes) /
                                             static_cast<double>(total_nodes);
                std::cout << "    (" << total_nodes << " nodes across "
                          << metrics_out.graph_breakdown.size()
                          << " partitions, avg " << (avg_graph_per_node / 1024.0)
                          << " KB per node)" << std::endl;
            }
            if (cfg.dump_index_sizes && !metrics_out.graph_breakdown.empty()) {
                std::cout << "  - Graph breakdown:" << std::endl;
                for (const auto& [name, stats] : metrics_out.graph_breakdown) {
                    std::cout << "    * " << name
                              << ": nodes=" << stats.nodes
                              << ", neighbors=" << (stats.neighbor_bytes / 1024.0 / 1024.0) << " MB"
                              << ", offsets=" << (stats.offsets_bytes / 1024.0 / 1024.0) << " MB"
                              << ", levels=" << (stats.levels_bytes / 1024.0 / 1024.0) << " MB"
                              << std::endl;
                }
            }
            std::cout.unsetf(std::ios_base::floatfield);
            std::cout << std::setprecision(6);

            std::vector<double> query_times;
            std::vector<double> recalls;
            std::vector<long long> all_cycles, all_instr, all_cache_ref, all_cache_miss, all_l1d_loads, all_l1d_miss, all_l1i_loads, all_l1i_miss, all_llc_loads, all_llc_miss;
            std::vector<long long> all_stall_fe, all_stall_be, all_br_miss, all_ctx_sw, all_cpu_mig, all_page_fault;

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                pqxx::connection c(conn_info);
                pqxx::work t(c);
                auto res = t.exec("SELECT role_id FROM UserRoles WHERE user_id = " + t.quote(q.user_id));
                t.commit();

                std::vector<std::pair<int,int>> all_results;
                double total_partition_time = 0.0;

                // Setup perf counters
                struct perf_event_attr pe;
                memset(&pe, 0, sizeof(pe));
                pe.size = sizeof(pe); pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
                pe.type = PERF_TYPE_HARDWARE;
                pe.config = PERF_COUNT_HW_CPU_CYCLES; int fd_cycles = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_INSTRUCTIONS; int fd_instr = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_CACHE_REFERENCES; int fd_cache_ref = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_CACHE_MISSES; int fd_cache_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND; int fd_stall_fe = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND; int fd_stall_be = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_HW_BRANCH_MISSES; int fd_br_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.type = PERF_TYPE_SOFTWARE;
                pe.config = PERF_COUNT_SW_CONTEXT_SWITCHES; int fd_ctx_sw = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_SW_CPU_MIGRATIONS; int fd_cpu_mig = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = PERF_COUNT_SW_PAGE_FAULTS; int fd_page_fault = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.type = PERF_TYPE_HW_CACHE;
                pe.config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS<<16); int fd_l1d_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16); int fd_l1d_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = (PERF_COUNT_HW_CACHE_L1I)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS<<16); int fd_l1i_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = (PERF_COUNT_HW_CACHE_L1I)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16); int fd_l1i_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS<<16); int fd_llc_loads = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                pe.config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16); int fd_llc_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
                int fds[] = {fd_cycles, fd_instr, fd_cache_ref, fd_cache_miss, fd_l1d_loads, fd_l1d_miss, fd_l1i_loads, fd_l1i_miss, fd_llc_loads, fd_llc_miss, fd_stall_fe, fd_stall_be, fd_br_miss, fd_ctx_sw, fd_cpu_mig, fd_page_fault};
                for (int fd : fds) if (fd >= 0) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }

                for (const auto& row : res) {
                    int role_id = row[0].as<int>();
                    std::string part_name = "documentblocks_role_" + std::to_string(role_id);
                    std::filesystem::path partition_dir = independent_dir / part_name;

                    {  // Scope for immediate index release
                        IndependentHNSWIndex idx(dimension, cfg.M, cfg.efConstruction);
                        if (!idx.load_components(partition_dir.string())) {
                            idx.build_from_partition(conn_info, part_name);
                            idx.set_ef_search(cfg.efSearch);
                            if (!cfg.save_index_dir.empty()) {
                                idx.save_components(partition_dir.string());
                            }
                        }
                        idx.set_ef_search(cfg.efSearch);

                        // Warm up
                        idx.search(q.query_vector.data(), q.topk);

                        // Timed search
                        auto start = std::chrono::high_resolution_clock::now();
                        auto results = idx.search(q.query_vector.data(), q.topk);
                        auto end = std::chrono::high_resolution_clock::now();

                        total_partition_time += std::chrono::duration<double, std::milli>(end - start).count();
                        all_results.insert(all_results.end(), results.begin(), results.end());
                    }  // idx released here, vectors and graph memory freed immediately
                }

                // Disable and read counters
                for (int fd : fds) if (fd >= 0) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
                long long cycles = 0, instr = 0, cache_ref = 0, cache_miss = 0, l1d_loads = 0, l1d_miss = 0, l1i_loads = 0, l1i_miss = 0, llc_loads = 0, llc_miss = 0;
                long long stall_fe = 0, stall_be = 0, br_miss = 0, ctx_sw = 0, cpu_mig = 0, page_fault = 0;
                if (fd_cycles >= 0) read(fd_cycles, &cycles, sizeof(cycles));
                if (fd_instr >= 0) read(fd_instr, &instr, sizeof(instr));
                if (fd_cache_ref >= 0) read(fd_cache_ref, &cache_ref, sizeof(cache_ref));
                if (fd_cache_miss >= 0) read(fd_cache_miss, &cache_miss, sizeof(cache_miss));
                if (fd_l1d_loads >= 0) read(fd_l1d_loads, &l1d_loads, sizeof(l1d_loads));
                if (fd_l1d_miss >= 0) read(fd_l1d_miss, &l1d_miss, sizeof(l1d_miss));
                if (fd_l1i_loads >= 0) read(fd_l1i_loads, &l1i_loads, sizeof(l1i_loads));
                if (fd_l1i_miss >= 0) read(fd_l1i_miss, &l1i_miss, sizeof(l1i_miss));
                if (fd_llc_loads >= 0) read(fd_llc_loads, &llc_loads, sizeof(llc_loads));
                if (fd_llc_miss >= 0) read(fd_llc_miss, &llc_miss, sizeof(llc_miss));
                if (fd_stall_fe >= 0) read(fd_stall_fe, &stall_fe, sizeof(stall_fe));
                if (fd_stall_be >= 0) read(fd_stall_be, &stall_be, sizeof(stall_be));
                if (fd_br_miss >= 0) read(fd_br_miss, &br_miss, sizeof(br_miss));
                if (fd_ctx_sw >= 0) read(fd_ctx_sw, &ctx_sw, sizeof(ctx_sw));
                if (fd_cpu_mig >= 0) read(fd_cpu_mig, &cpu_mig, sizeof(cpu_mig));
                if (fd_page_fault >= 0) read(fd_page_fault, &page_fault, sizeof(page_fault));

                all_cycles.push_back(cycles); all_instr.push_back(instr); all_cache_ref.push_back(cache_ref); all_cache_miss.push_back(cache_miss);
                all_l1d_loads.push_back(l1d_loads); all_l1d_miss.push_back(l1d_miss); all_l1i_loads.push_back(l1i_loads); all_l1i_miss.push_back(l1i_miss);
                all_llc_loads.push_back(llc_loads); all_llc_miss.push_back(llc_miss);
                all_stall_fe.push_back(stall_fe); all_stall_be.push_back(stall_be); all_br_miss.push_back(br_miss);
                all_ctx_sw.push_back(ctx_sw); all_cpu_mig.push_back(cpu_mig); all_page_fault.push_back(page_fault);

                if (i == 0) {
                    std::cout << "\n=== Perf Stats (First Query) ===" << std::endl;
                    std::cout << "Cycles: " << cycles << std::endl;
                    std::cout << "Instructions: " << instr << std::endl;
                    std::cout << "IPC: " << (cycles > 0 ? (double)instr/cycles : 0.0) << std::endl;
                    std::cout << "Stalled cycles (frontend): " << stall_fe << " (" << (cycles > 0 ? (double)stall_fe*100.0/cycles : 0.0) << "%)" << std::endl;
                    std::cout << "Stalled cycles (backend): " << stall_be << " (" << (cycles > 0 ? (double)stall_be*100.0/cycles : 0.0) << "%)" << std::endl;
                    std::cout << "Branch misses: " << br_miss << std::endl;
                    std::cout << "Context switches: " << ctx_sw << std::endl;
                    std::cout << "CPU migrations: " << cpu_mig << std::endl;
                    std::cout << "Page faults: " << page_fault << std::endl;
                    std::cout << "Cache refs: " << cache_ref << ", misses: " << cache_miss
                              << " (" << (cache_ref > 0 ? (double)cache_miss*100.0/cache_ref : 0.0) << "%)" << std::endl;
                    std::cout << "L1D loads: " << l1d_loads << ", misses: " << l1d_miss
                              << " (" << (l1d_loads > 0 ? (double)l1d_miss*100.0/l1d_loads : 0.0) << "%)" << std::endl;
                    std::cout << "L1I loads: " << l1i_loads << ", misses: " << l1i_miss
                              << " (" << (l1i_loads > 0 ? (double)l1i_miss*100.0/l1i_loads : 0.0) << "%)" << std::endl;
                    std::cout << "LLC loads: " << llc_loads << ", misses: " << llc_miss
                              << " (" << (llc_loads > 0 ? (double)llc_miss*100.0/llc_loads : 0.0) << "%)" << std::endl;
                    std::cout << "================================" << std::endl;
                }

                for (int fd : fds) if (fd >= 0) close(fd);

                query_times.push_back(total_partition_time);

                // Debug: print first query's results
                if (i == 0) {
                    std::cout << "\n=== DEBUG: First Query (Independent HNSW) ===" << std::endl;
                    std::cout << "Search results (first 3):" << std::endl;
                    for (size_t j = 0; j < std::min(size_t(3), all_results.size()); j++) {
                        std::cout << "  [" << j << "] doc_id=" << all_results[j].first
                                  << ", block_id=" << all_results[j].second << std::endl;
                    }
                    std::cout << "Ground truth (first 3):" << std::endl;
                    for (size_t j = 0; j < std::min(size_t(3), ground_truth[i].size()); j++) {
                        std::cout << "  [" << j << "] doc_id=" << ground_truth[i][j].first
                                  << ", block_id=" << ground_truth[i][j].second << std::endl;
                    }
                    std::cout << "========================\n" << std::endl;
                }

                recalls.push_back(calculate_recall(all_results, ground_truth[i]));
            }

            metrics_out.avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                                    query_times.size();
            metrics_out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                      recalls.size();
            auto indep_times_copy = query_times;
            metrics_out.p50 = percentile(indep_times_copy, 0.5);
            metrics_out.p90 = percentile(indep_times_copy, 0.9);
            metrics_out.p95 = percentile(indep_times_copy, 0.95);
            metrics_out.p99 = percentile(indep_times_copy, 0.99);

            query_times_out = query_times;

            std::cout << "Results:" << std::endl;
            std::cout << "  Avg time: " << metrics_out.avg_time << " ms" << std::endl;
            std::cout << "  P50: " << metrics_out.p50 << " ms, P90: " << metrics_out.p90
                      << " ms, P95: " << metrics_out.p95 << " ms, P99: " << metrics_out.p99 << " ms"
                      << std::endl;
            std::cout << "  Avg recall: " << metrics_out.avg_recall << std::endl;

            // Print average perf stats
            auto avg = [](const std::vector<long long>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0LL) / (double)v.size(); };
            double avg_cycles = avg(all_cycles), avg_instr = avg(all_instr), avg_cache_ref = avg(all_cache_ref);
            double avg_cache_miss = avg(all_cache_miss), avg_l1d_loads = avg(all_l1d_loads), avg_l1d_miss = avg(all_l1d_miss);
            double avg_l1i_loads = avg(all_l1i_loads), avg_l1i_miss = avg(all_l1i_miss);
            double avg_llc_loads = avg(all_llc_loads), avg_llc_miss = avg(all_llc_miss);
            double avg_stall_fe = avg(all_stall_fe), avg_stall_be = avg(all_stall_be), avg_br_miss = avg(all_br_miss);
            double avg_ctx_sw = avg(all_ctx_sw), avg_cpu_mig = avg(all_cpu_mig), avg_page_fault = avg(all_page_fault);
            std::cout << "\n=== Average Perf Stats (All Queries) ===" << std::endl;
            std::cout << "Avg Cycles: " << avg_cycles << std::endl;
            std::cout << "Avg Instructions: " << avg_instr << std::endl;
            std::cout << "Avg IPC: " << (avg_cycles > 0 ? avg_instr/avg_cycles : 0.0) << std::endl;
            std::cout << "Avg Stalled cycles (frontend): " << avg_stall_fe << " (" << (avg_cycles > 0 ? avg_stall_fe*100.0/avg_cycles : 0.0) << "%)" << std::endl;
            std::cout << "Avg Stalled cycles (backend): " << avg_stall_be << " (" << (avg_cycles > 0 ? avg_stall_be*100.0/avg_cycles : 0.0) << "%)" << std::endl;
            std::cout << "Avg Branch misses: " << avg_br_miss << std::endl;
            std::cout << "Avg Context switches: " << avg_ctx_sw << std::endl;
            std::cout << "Avg CPU migrations: " << avg_cpu_mig << std::endl;
            std::cout << "Avg Page faults: " << avg_page_fault << std::endl;
            std::cout << "Avg Cache refs: " << avg_cache_ref << ", misses: " << avg_cache_miss
                      << " (" << (avg_cache_ref > 0 ? avg_cache_miss*100.0/avg_cache_ref : 0.0) << "%)" << std::endl;
            std::cout << "Avg L1D loads: " << avg_l1d_loads << ", misses: " << avg_l1d_miss
                      << " (" << (avg_l1d_loads > 0 ? avg_l1d_miss*100.0/avg_l1d_loads : 0.0) << "%)" << std::endl;
            std::cout << "Avg L1I loads: " << avg_l1i_loads << ", misses: " << avg_l1i_miss
                      << " (" << (avg_l1i_loads > 0 ? avg_l1i_miss*100.0/avg_l1i_loads : 0.0) << "%)" << std::endl;
            std::cout << "Avg LLC loads: " << avg_llc_loads << ", misses: " << avg_llc_miss
                      << " (" << (avg_llc_loads > 0 ? avg_llc_miss*100.0/avg_llc_loads : 0.0) << "%)" << std::endl;
            std::cout << "========================================" << std::endl;
        };

        auto evaluate_global_runtime = [&](const std::string& title,
                                           const std::shared_ptr<SharedVectorTable>& shared_table_local,
                                           const std::filesystem::path& graph_path,
                                           const std::unordered_map<int, std::vector<faiss::idx_t>>& role_to_internal_ids,
                                           GlobalMetrics& metrics_out) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "SCENARIO 2: " << title << std::endl;
            std::cout << "========================================" << std::endl;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Memory: " << (metrics_out.total_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  - Stored vectors (shared): " << (metrics_out.vector_bytes / 1024.0 / 1024.0)
                      << " MB" << std::endl;
            std::cout << "  - Graph structure: " << (metrics_out.graph_bytes / 1024.0 / 1024.0)
                      << " MB" << std::endl;
            if (metrics_out.graph_stats.nodes > 0) {
                double avg_graph_per_node = static_cast<double>(metrics_out.graph_bytes) /
                                             static_cast<double>(metrics_out.graph_stats.nodes);
                std::cout << "    (" << metrics_out.graph_stats.nodes << " nodes, avg "
                          << (avg_graph_per_node / 1024.0) << " KB per node)" << std::endl;
            }
            std::cout.unsetf(std::ios_base::floatfield);
            std::cout << std::setprecision(6);

            std::vector<double> query_times;
            std::vector<double> recalls;
            std::vector<float> temp_dist;
            std::vector<faiss::idx_t> temp_idx;
            const auto ntotal_local = shared_table_local->get_index_flat()->ntotal;
            std::vector<uint8_t> mark(static_cast<size_t>(ntotal_local), 0);
            std::vector<faiss::idx_t> allowed_ids;
            std::vector<faiss::idx_t> touched_ids;

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                pqxx::connection c(conn_info);
                pqxx::work t(c);
                auto res = t.exec("SELECT role_id FROM UserRoles WHERE user_id = " + t.quote(q.user_id));
                t.commit();

                std::unordered_set<int> allowed_roles;
                for (const auto& row : res) {
                    allowed_roles.insert(row[0].as<int>());
                }

                allowed_ids.clear();
                touched_ids.clear();

                GlobalHNSWIndex idx(IndexMode::SHARED_POINTER, shared_table_local, cfg.M, cfg.efConstruction);
                if (!idx.load_graph(graph_path.string())) {
                    idx.build();
                    if (!cfg.save_index_dir.empty()) {
                        idx.save_graph(graph_path.string());
                    }
                }
                idx.set_ef_search(cfg.global_ef_search);

                for (int role_id : allowed_roles) {
                    auto role_it = role_to_internal_ids.find(role_id);
                    if (role_it == role_to_internal_ids.end()) {
                        continue;
                    }
                    for (faiss::idx_t internal_id : role_it->second) {
                        size_t pos = static_cast<size_t>(internal_id);
                        if (pos >= mark.size()) {
                            continue;
                        }
                        if (!mark[pos]) {
                            mark[pos] = 1;
                            touched_ids.push_back(internal_id);
                            allowed_ids.push_back(internal_id);
                        }
                    }
                }

                std::vector<SharedVectorTable::DocBlockId> filtered;
                double time_ms = 0.0;
                if (!allowed_ids.empty()) {
                    auto start = std::chrono::high_resolution_clock::now();
                    filtered = idx.search_filtered(
                        q.query_vector.data(),
                        cfg.topk,
                        allowed_ids,
                        mark,
                        temp_dist,
                        temp_idx);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                }

                for (faiss::idx_t id : touched_ids) {
                    size_t pos = static_cast<size_t>(id);
                    if (pos < mark.size()) {
                        mark[pos] = 0;
                    }
                }

                query_times.push_back(time_ms);
                recalls.push_back(calculate_recall(filtered, ground_truth[i]));
            }

            metrics_out.avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                                     query_times.size();
            metrics_out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                       recalls.size();
            metrics_out.p50 = percentile(query_times, 0.5);
            metrics_out.p90 = percentile(query_times, 0.9);
            metrics_out.p95 = percentile(query_times, 0.95);
            metrics_out.p99 = percentile(query_times, 0.99);

            std::cout << "Results:" << std::endl;
            std::cout << "  Avg time: " << metrics_out.avg_time << " ms" << std::endl;
            std::cout << "  P50: " << metrics_out.p50 << " ms, P90: " << metrics_out.p90
                      << " ms, P95: " << metrics_out.p95 << " ms, P99: " << metrics_out.p99 << " ms"
                      << std::endl;
            std::cout << "  Avg recall: " << metrics_out.avg_recall << std::endl;
        };

        ensure_pointer_graphs(false, pointer_shared_dir, shared_table);
        if (cfg.pointer_copy_baseline) {
            ensure_pointer_graphs(true, pointer_copy_dir, shared_table);
        }
        ensure_global_graph(global_dir, shared_table);
        ensure_independent_components(independent_dir);

        std::unordered_map<SharedVectorTable::DocBlockId, std::vector<int>, SharedVectorTable::DocBlockIdHash> doc_roles_map;

        PointerMetrics pointer_metrics;
        std::vector<double> pointer_query_times;
        collect_pointer_stats(pointer_shared_dir, shared_table, pointer_metrics, &doc_roles_map);

        std::unordered_map<int, std::vector<faiss::idx_t>> role_to_internal_ids;
        role_to_internal_ids.reserve(doc_roles_map.size());
        for (const auto& [doc_block, roles] : doc_roles_map) {
            faiss::idx_t idx = shared_table->get_internal_index(doc_block.first, doc_block.second);
            if (idx < 0) {
                continue;
            }
            for (int role : roles) {
                role_to_internal_ids[role].push_back(idx);
            }
        }
        for (auto& [role, ids] : role_to_internal_ids) {
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        }
        evaluate_pointer_runtime("Pointer HNSW (Shared Storage)",
                                 shared_table,
                                 pointer_shared_dir,
                                 pointer_metrics,
                                 pointer_query_times);

        if (cfg.pointer_copy_baseline) {
            PointerMetrics pointer_copy_metrics;
            std::vector<double> pointer_copy_query_times;
            collect_pointer_stats(pointer_copy_dir, shared_table, pointer_copy_metrics, nullptr);
            evaluate_pointer_runtime("Pointer HNSW (Legacy Copy Mode)",
                                     shared_table,
                                     pointer_copy_dir,
                                     pointer_copy_metrics,
                                     pointer_copy_query_times);
        }

        // GlobalMetrics global_metrics;
        // std::filesystem::path global_graph_path = global_dir / "global_graph.bin";
        // collect_global_stats(global_graph_path, shared_table, global_metrics);
        // evaluate_global_runtime("Global HNSW (Post-filter)",
        //                         shared_table,
        //                         global_graph_path,
        //                         doc_roles_map,
        //                         global_metrics);

        PointerMetrics independent_metrics;
        std::vector<double> independent_query_times;
        collect_independent_stats(independent_dir, independent_metrics);
        evaluate_independent_runtime("Independent HNSW (Baseline)",
                                     independent_dir,
                                     independent_metrics,
                                     independent_query_times);

        // ===================================================================
        // COMPARISON
        // ===================================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "COMPARISON: Pointer vs Global vs Independent" << std::endl;
        std::cout << "========================================" << std::endl;

        double pointer_total_mb = pointer_metrics.total_bytes() / 1024.0 / 1024.0;
        double pointer_vector_mb = pointer_metrics.vector_bytes / 1024.0 / 1024.0;
        double pointer_graph_mb = pointer_metrics.graph_bytes / 1024.0 / 1024.0;
        // double global_total_mb = global_metrics.total_bytes() / 1024.0 / 1024.0;
        // double global_vector_mb = global_metrics.vector_bytes / 1024.0 / 1024.0;
        // double global_graph_mb = global_metrics.graph_bytes / 1024.0 / 1024.0;
        double independent_total_mb = independent_metrics.total_bytes() / 1024.0 / 1024.0;
        double independent_vector_mb = independent_metrics.vector_bytes / 1024.0 / 1024.0;
        double independent_graph_mb = independent_metrics.graph_bytes / 1024.0 / 1024.0;

        double pointer_mem_savings_mb = independent_total_mb - pointer_total_mb;
        double pointer_mem_savings_pct =
                (1.0 - static_cast<double>(pointer_metrics.total_bytes()) /
                               static_cast<double>(independent_metrics.total_bytes())) *
                100.0;

        // double global_mem_savings_mb = independent_total_mb - global_total_mb;
        // double global_mem_savings_pct =
        //         (1.0 - static_cast<double>(global_metrics.total_bytes()) /
        //                        static_cast<double>(independent_metrics.total_bytes())) *
        //         100.0;

        double pointer_recall_diff = std::abs(pointer_metrics.avg_recall - independent_metrics.avg_recall);
        // double global_recall_diff = std::abs(global_metrics.avg_recall - independent_metrics.avg_recall);
        double pointer_slowdown = pointer_metrics.avg_time / independent_metrics.avg_time;
        // double global_slowdown = global_metrics.avg_time / independent_metrics.avg_time;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nMemory:" << std::endl;
        std::cout << "  Pointer (shared vectors):    " << pointer_total_mb << " MB (vectors "
                  << pointer_vector_mb << " MB, graph " << pointer_graph_mb << " MB)" << std::endl;
        // std::cout << "  Global (post-filter HNSW):  " << global_total_mb << " MB (vectors "
        //           << global_vector_mb << " MB, graph " << global_graph_mb << " MB)" << std::endl;
        std::cout << "  Independent:                " << independent_total_mb << " MB (vectors "
                  << independent_vector_mb << " MB, graph " << independent_graph_mb << " MB)" << std::endl;
        std::cout << "  Pointer vs Independent savings:  " << pointer_mem_savings_mb << " MB ("
                  << pointer_mem_savings_pct << "%)" << std::endl;
        // std::cout << "  Global vs Independent savings:   " << global_mem_savings_mb << " MB ("
        //           << global_mem_savings_pct << "%)" << std::endl;

        std::cout << std::setprecision(4);
        std::cout << "\nRecall (avg):" << std::endl;
        std::cout << "  Pointer:      " << pointer_metrics.avg_recall << " (diff vs independent "
                  << pointer_recall_diff << ")" << std::endl;
        // std::cout << "  Global:       " << global_metrics.avg_recall << " (diff vs independent "
        //           << global_recall_diff << ")" << std::endl;
        std::cout << "  Independent:  " << independent_metrics.avg_recall << std::endl;

        std::cout << std::setprecision(2);
        std::cout << "\nSearch Speed (avg):" << std::endl;
        std::cout << "  Pointer:      " << pointer_metrics.avg_time << " ms/query" << std::endl;
        // std::cout << "  Global:       " << global_metrics.avg_time << " ms/query" << std::endl;
        std::cout << "  Independent:  " << independent_metrics.avg_time << " ms/query" << std::endl;
        std::cout << "  Pointer slowdown vs independent: " << pointer_slowdown << "x ("
                  << ((pointer_slowdown - 1.0) * 100) << "% slower)" << std::endl;
        // std::cout << "  Global slowdown vs independent:  " << global_slowdown << "x ("
        //           << ((global_slowdown - 1.0) * 100) << "% slower)" << std::endl;

        std::cout << "\nSearch Speed (percentiles):" << std::endl;
        std::cout << "  P50: Pointer=" << pointer_metrics.p50 << "ms, Independent=" << independent_metrics.p50 << "ms" << std::endl;
        std::cout << "  P90: Pointer=" << pointer_metrics.p90 << "ms, Independent=" << independent_metrics.p90 << "ms" << std::endl;
        std::cout << "  P95: Pointer=" << pointer_metrics.p95 << "ms, Independent=" << independent_metrics.p95 << "ms" << std::endl;
        std::cout << "  P99: Pointer=" << pointer_metrics.p99 << "ms, Independent=" << independent_metrics.p99 << "ms" << std::endl;

        // std::vector<std::pair<std::string, HNSWGraphStats>> global_graph_details = {
        //         {"global", global_metrics.graph_stats}};

        json shared_breakdown_json = graph_breakdown_to_json(pointer_metrics.graph_breakdown);
        // json global_breakdown_json = graph_breakdown_to_json(global_graph_details);
        json independent_breakdown_json = graph_breakdown_to_json(independent_metrics.graph_breakdown);

        // Save comparison results
        json output = {
            {"config", {
                {"M", cfg.M},
                {"efConstruction", cfg.efConstruction},
                {"efSearch", cfg.efSearch},
                {"topk", cfg.topk},
                {"num_queries", queries.size()},
                {"num_partitions", partitions.size()}
            }},
            {"shared_storage", {
                {"total_memory_mb", pointer_total_mb},
                {"vector_memory_mb", pointer_vector_mb},
                {"graph_memory_mb", pointer_graph_mb},
                {"avg_time_ms", pointer_metrics.avg_time},
                {"p50_ms", pointer_metrics.p50},
                {"p90_ms", pointer_metrics.p90},
                {"p95_ms", pointer_metrics.p95},
                {"p99_ms", pointer_metrics.p99},
                {"avg_recall", pointer_metrics.avg_recall},
                {"graph_breakdown", shared_breakdown_json}
            }},
            // {"global_storage", {
            //     {"total_memory_mb", global_total_mb},
            //     {"vector_memory_mb", global_vector_mb},
            //     {"graph_memory_mb", global_graph_mb},
            //     {"avg_time_ms", global_metrics.avg_time},
            //     {"p50_ms", global_metrics.p50},
            //     {"p90_ms", global_metrics.p90},
            //     {"p95_ms", global_metrics.p95},
            //     {"p99_ms", global_metrics.p99},
            //     {"avg_recall", global_metrics.avg_recall},
            //     {"graph_breakdown", global_breakdown_json}
            // }},
            {"independent_storage", {
                {"total_memory_mb", independent_total_mb},
                {"vector_memory_mb", independent_vector_mb},
                {"graph_memory_mb", independent_graph_mb},
                {"avg_time_ms", independent_metrics.avg_time},
                {"p50_ms", independent_metrics.p50},
                {"p90_ms", independent_metrics.p90},
                {"p95_ms", independent_metrics.p95},
                {"p99_ms", independent_metrics.p99},
                {"avg_recall", independent_metrics.avg_recall},
                {"graph_breakdown", independent_breakdown_json}
            }},
            {"comparison", {
                {"pointer_vs_independent", {
                    {"memory_savings_mb", pointer_mem_savings_mb},
                    {"memory_savings_percent", pointer_mem_savings_pct},
                    {"avg_time_ratio", pointer_slowdown},
                    {"avg_time_percent_slower", (pointer_slowdown - 1.0) * 100},
                    {"avg_recall_diff", pointer_recall_diff}
                }},
                // {"global_vs_independent", {
            //     {"memory_savings_mb", global_mem_savings_mb},
            //     {"memory_savings_percent", global_mem_savings_pct},
            //     {"avg_time_ratio", global_slowdown},
            //     {"avg_time_percent_slower", (global_slowdown - 1.0) * 100},
            //     {"avg_recall_diff", global_recall_diff}
            // }},
                {"recall_note",
                 "Ground truth computation skipped (too slow); values shown are based on filtered results"}
            }}
        };

        std::filesystem::path output_dir = project_root / "logical_partition_benchmark/benchmark/results";
        std::filesystem::create_directories(output_dir);

        std::filesystem::path times_csv = output_dir / "query_times.csv";
        bool csv_written = false;
        {
            std::ofstream csv(times_csv);
            if (!csv) {
                std::cerr << "Warning: failed to open " << times_csv << " for writing query times" << std::endl;
            } else {
                csv << "query_index,pointer_ms,independent_ms\n";
                for (size_t i = 0; i < queries.size(); ++i) {
                    double pointer_ms = i < pointer_query_times.size() ? pointer_query_times[i] : 0.0;
                    double independent_ms = i < independent_query_times.size() ? independent_query_times[i] : 0.0;
                    csv << i << ',' << pointer_ms << ',' << independent_ms << '\n';
                }
                csv_written = true;
            }
        }

        if (csv_written) {
            std::filesystem::path pdf_path = output_dir / "query_times.pdf";
            std::ostringstream plot_cmd;
            plot_cmd << "python3 - <<'PY'\n";
            plot_cmd << "import csv\n";
            plot_cmd << "from pathlib import Path\n";
            plot_cmd << "import matplotlib\n";
            plot_cmd << "matplotlib.use('Agg')\n";
            plot_cmd << "import matplotlib.pyplot as plt\n";
            plot_cmd << "csv_path = Path(r\"\"\"" << times_csv.string() << "\"\"\")\n";
            plot_cmd << "pdf_path = Path(r\"\"\"" << pdf_path.string() << "\"\"\")\n";
            plot_cmd << "pointer = []\n";
            plot_cmd << "independent = []\n";
            plot_cmd << "with csv_path.open() as f:\n";
            plot_cmd << "    reader = csv.reader(f)\n";
            plot_cmd << "    next(reader, None)\n";
            plot_cmd << "    for row in reader:\n";
            plot_cmd << "        if len(row) < 3:\n";
            plot_cmd << "            continue\n";
            plot_cmd << "        pointer.append(float(row[1]))\n";
            plot_cmd << "        independent.append(float(row[2]))\n";
            plot_cmd << "fig, ax = plt.subplots(figsize=(8, 4.5))\n";
            plot_cmd << "if pointer:\n";
            plot_cmd << "    ax.plot(range(len(pointer)), pointer, label='Pointer', linewidth=1.0)\n";
            plot_cmd << "if independent:\n";
            plot_cmd << "    ax.plot(range(len(independent)), independent, label='Independent', linewidth=1.0)\n";
            plot_cmd << "ax.set_xlabel('Query Index')\n";
            plot_cmd << "ax.set_ylabel('Latency (ms)')\n";
            plot_cmd << "ax.set_title('Query Latency Distribution')\n";
            plot_cmd << "ax.legend()\n";
            plot_cmd << "ax.grid(True, alpha=0.3)\n";
            plot_cmd << "fig.tight_layout()\n";
            plot_cmd << "fig.savefig(pdf_path)\n";
            plot_cmd << "plt.close(fig)\n";
            plot_cmd << "PY\n";

            int plot_status = std::system(plot_cmd.str().c_str());
            if (plot_status != 0) {
                std::cerr << "Warning: failed to generate query time plot (python exit code "
                          << plot_status << ")" << std::endl;
            } else {
                std::cout << "✓ Query time plot saved to " << pdf_path << std::endl;
            }
        }

        std::string output_file = (output_dir / "comparison_results.json").string();
        std::ofstream out(output_file);
        out << output.dump(4);
        out.close();

        std::cout << "\n✓ Results saved to " << output_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
