#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pqxx/pqxx>

#include "benchmark_utils.h"
#include "global_hnsw_index.h"

using pointer_benchmark::GlobalHNSWIndex;
using pointer_benchmark::IndexMode;
using pointer_benchmark::SharedVectorTable;

namespace {
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
        std::cout << "Test: Global HNSW - Post-filter (Physical)" << std::endl;
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

        const int globalEfSearch = parse_ef_search(argc, argv, 300);
        const int warmup_iterations = parse_warmup_iterations(argc, argv, 0);
        std::cout << "ef_search: " << globalEfSearch << std::endl;
        std::cout << "warmup iterations: " << warmup_iterations << std::endl;

        const std::filesystem::path base_dir = benchmark_utils::load_index_storage_path();
        std::cout << "Index storage path: " << base_dir << std::endl;
        const auto postfilter_dir = base_dir / "physical_postfilter";
        std::filesystem::create_directories(postfilter_dir);
        const auto shared_cache = postfilter_dir / "shared_vectors.bin";
        const auto global_graph_path = postfilter_dir / "global_graph.bin";

        auto shared_table = benchmark_utils::load_shared_table(dimension, shared_cache, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        GlobalHNSWIndex global_index(IndexMode::INDEPENDENT, shared_table, M, efConstruction);
        if (!std::filesystem::exists(global_graph_path) ||
            !global_index.load_graph(global_graph_path.string())) {
            std::cout << "Building global HNSW graph..." << std::endl;
            global_index.build();
            global_index.save_graph(global_graph_path.string());
        } else {
            std::cout << "Loaded global graph from disk" << std::endl;
        }
        global_index.set_ef_search(globalEfSearch);

        const size_t ntotal = global_index.ntotal();

        // For INDEPENDENT mode, memory_bytes() includes both vectors and graph
        const size_t total_bytes = global_index.memory_bytes();
        const size_t vector_bytes = ntotal * dimension * sizeof(float);
        const size_t graph_bytes = (total_bytes > vector_bytes) ? (total_bytes - vector_bytes) : 0;

        benchmark_utils::HNSWGraphStats graph_stats{};
        graph_stats.nodes = ntotal;

        std::cout << "\n--- Storage ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Vectors: " << (vector_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Graph: " << (graph_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Nodes: " << graph_stats.nodes << std::endl;
        std::cout << "Total storage: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

        auto user_roles = benchmark_utils::load_user_roles(conn_info);
        auto doc_roles = benchmark_utils::load_document_roles(conn_info);
        auto doc_block_roles = benchmark_utils::load_doc_block_roles(conn_info);
        bool ground_truth_from_cache = false;
        auto ground_truth = benchmark_utils::load_or_compute_ground_truth(
            queries, shared_table, user_roles, doc_roles, &ground_truth_from_cache);
        if (ground_truth_from_cache) {
            std::cout << "Loaded ground truth from cache" << std::endl;
        } else {
            std::cout << "Computed ground truth and saved to cache" << std::endl;
        }

        std::unordered_map<int, std::vector<faiss::idx_t>> role_to_internal_ids;
        role_to_internal_ids.reserve(doc_block_roles.size());
        for (const auto& [doc_block, roles] : doc_block_roles) {
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

        std::vector<double> query_times;
        std::vector<double> recalls;
        query_times.reserve(queries.size());
        recalls.reserve(queries.size());
        std::vector<float> temp_dist;
        std::vector<faiss::idx_t> temp_idx;
        const auto ntotal_val = shared_table->get_index_flat()->ntotal;
        std::vector<uint8_t> mark(static_cast<size_t>(ntotal_val), 0);
        std::vector<faiss::idx_t> allowed_ids;
        std::vector<faiss::idx_t> touched_ids;

        auto build_allowed = [&](const std::vector<int>& roles,
                                std::vector<faiss::idx_t>& allowed,
                                std::vector<faiss::idx_t>& touched) {
            allowed.clear();
            touched.clear();
            for (int role_id : roles) {
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
                        touched.push_back(internal_id);
                        allowed.push_back(internal_id);
                    }
                }
            }
        };

        auto reset_marks = [&](const std::vector<faiss::idx_t>& touched) {
            for (faiss::idx_t id : touched) {
                size_t pos = static_cast<size_t>(id);
                if (pos < mark.size()) {
                    mark[pos] = 0;
                }
            }
        };

        std::cout << "\n--- Query Performance ---" << std::endl;
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto& query = queries[i];

            std::vector<int> query_roles;
            if (auto cache_it = user_roles.find(query.user_id); cache_it != user_roles.end()) {
                query_roles.assign(cache_it->second.begin(), cache_it->second.end());
            }

            for (int warm = 0; warm < warmup_iterations; ++warm) {
                build_allowed(query_roles, allowed_ids, touched_ids);
                if (allowed_ids.empty()) {
                    break;
                }
                auto warm_results = global_index.search_filtered(
                    query.query_vector.data(),
                    query.topk,
                    allowed_ids,
                    mark,
                    temp_dist,
                    temp_idx);
                (void)warm_results;
                reset_marks(touched_ids);
            }

            build_allowed(query_roles, allowed_ids, touched_ids);

            std::vector<SharedVectorTable::DocBlockId> results;
            double time_ms = 0.0;
            if (!allowed_ids.empty()) {
                auto start = std::chrono::high_resolution_clock::now();
                results = global_index.search_filtered(
                    query.query_vector.data(),
                    query.topk,
                    allowed_ids,
                    mark,
                    temp_dist,
                    temp_idx);
                auto end = std::chrono::high_resolution_clock::now();
                time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            }

            reset_marks(touched_ids);

            if (i == 0) {
                std::cout << "\n=== Debug: Postfilter first query ===" << std::endl;
                std::cout << "Results count: " << results.size() << std::endl;
                for (size_t j = 0; j < std::min<size_t>(results.size(), 5); ++j) {
                    std::cout << "  result[" << j << "]: doc=" << results[j].first
                              << ", block=" << results[j].second << std::endl;
                }
                const auto& gt = ground_truth[i];
                std::cout << "Ground truth count: " << gt.size() << std::endl;
                for (size_t j = 0; j < std::min<size_t>(gt.size(), 5); ++j) {
                    std::cout << "  gt[" << j << "]: doc=" << gt[j].first
                              << ", block=" << gt[j].second << std::endl;
                }
                size_t hits = 0;
                for (const auto& doc_block : results) {
                    if (std::find(gt.begin(), gt.end(), doc_block) != gt.end()) {
                        ++hits;
                    }
                }
                std::cout << "Matches with ground truth: " << hits << std::endl;
                std::cout << "===============================\n" << std::endl;
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

        benchmark_utils::json output = {
            {"strategy", "physical_postfilter"},
            {"storage", {
                {"vector_bytes", vector_bytes},
                {"graph_bytes", graph_bytes},
                {"total_bytes", total_bytes},
                {"vector_mb", vector_bytes / 1024.0 / 1024.0},
                {"graph_mb", graph_bytes / 1024.0 / 1024.0},
                {"total_mb", total_bytes / 1024.0 / 1024.0},
                {"graph_nodes", graph_stats.nodes}
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
                {"efSearch", globalEfSearch}
            }}
        };

        const auto output_path = root / "logical_partition_benchmark/benchmark/src/physical_postfilter_results.json";
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
