#include <algorithm>
#include <chrono>
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
#include "pointer_hnsw_index.h"

using pointer_benchmark::PointerHNSWIndex;
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
        std::cout << "Test: Pointer HNSW - Role Partition" << std::endl;
        std::cout << "========================================\n" << std::endl;

        const auto root = benchmark_utils::project_root();
        const auto conn_info = benchmark_utils::load_connection_string(root / "config.json");
        auto queries = benchmark_utils::load_queries(root / "basic_benchmark/query_dataset.json");
        if (queries.empty()) {
            throw std::runtime_error("Query dataset is empty");
        }
        std::cout << "Loaded " << queries.size() << " queries" << std::endl;

        const int dimension = static_cast<int>(queries.front().query_vector.size());

        // Load HNSW config
        auto hnsw_config = benchmark_utils::load_hnsw_config();
        const int M = hnsw_config.M;
        const int efConstruction = hnsw_config.ef_construction;
        std::cout << "HNSW config: M=" << M << ", ef_construction=" << efConstruction << std::endl;

        const int efSearch = parse_ef_search(argc, argv, 15);
        const int warmup_iterations = parse_warmup_iterations(argc, argv, 0);
        std::cout << "ef_search: " << efSearch << std::endl;
        std::cout << "warmup iterations: " << warmup_iterations << std::endl;

        const std::filesystem::path base_dir = benchmark_utils::load_index_storage_path();
        std::cout << "Index storage path: " << base_dir << std::endl;
        const auto role_dir = base_dir / "role_partition";
        std::filesystem::create_directories(role_dir);
        const auto shared_cache = role_dir / "shared_vectors.bin";

        auto shared_table = benchmark_utils::load_shared_table(dimension, shared_cache, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        const auto partitions = benchmark_utils::fetch_tables_with_prefix(conn_info, "documentblocks_role_");
        std::cout << "Found " << partitions.size() << " role partitions" << std::endl;

        std::unordered_map<int, std::unique_ptr<PointerHNSWIndex>> indexes;
        std::vector<std::pair<std::string, benchmark_utils::HNSWGraphStats>> graph_breakdown;
        size_t total_graph_bytes = 0;

        for (const auto& table : partitions) {
            const auto suffix_pos = table.find_last_of('_');
            if (suffix_pos == std::string::npos) {
                continue;
            }
            const int role_id = std::stoi(table.substr(suffix_pos + 1));
            auto graph_path = role_dir / (table + "_graph.bin");
            auto index = std::make_unique<PointerHNSWIndex>(shared_table, M, efConstruction);
            if (!std::filesystem::exists(graph_path) || !index->load_graph(graph_path.string())) {
                std::cout << "  Building " << table << "..." << std::endl;
                index->build_from_partition(conn_info, table);
                index->save_graph(graph_path.string());
            } else {
                std::cout << "  Loaded " << table << " from disk" << std::endl;
            }

            index->set_ef_search(efSearch);
            auto stats = benchmark_utils::compute_hnsw_graph_stats(index->get_hnsw_index()->hnsw);
            total_graph_bytes += stats.total_bytes();
            graph_breakdown.emplace_back(table, stats);

            std::cout << "    " << index->size() << " vectors, graph="
                      << (stats.total_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;

            indexes.emplace(role_id, std::move(index));
        }

        const size_t vector_bytes = shared_table->get_index_flat()->ntotal * dimension * sizeof(float);
        const size_t total_bytes = vector_bytes + total_graph_bytes;

        std::cout << "\n--- Storage ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Shared vectors: " << (vector_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Total graphs: " << (total_graph_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
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
                    auto cached = user_roles.find(query.user_id);
                    if (cached != user_roles.end()) {
                        query_roles.assign(cached->second.begin(), cached->second.end());
                    }
                }
            }

            std::vector<SharedVectorTable::DocBlockId> results;
            std::unordered_set<SharedVectorTable::DocBlockId, SharedVectorTable::DocBlockIdHash> seen;
            double total_partition_time = 0.0;

            if (!query_roles.empty()) {
                for (int role_id : query_roles) {
                    auto idx_it = indexes.find(role_id);
                    if (idx_it == indexes.end()) {
                        continue;
                    }

                    for (int warm = 0; warm < warmup_iterations; ++warm) {
                        idx_it->second->search(query.query_vector.data(), query.topk);
                    }

                    auto start = std::chrono::high_resolution_clock::now();
                    auto hits = idx_it->second->search(query.query_vector.data(), query.topk);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_partition_time += std::chrono::duration<double, std::milli>(end - start).count();

                    for (const auto& doc_block : hits) {
                        if (seen.insert(doc_block).second) {
                            results.push_back(doc_block);
                        }
                    }
                }
            }

            query_times.push_back(total_partition_time);
            recalls.push_back(benchmark_utils::calculate_recall(results, ground_truth[i]));

            if (i == 0) {
                std::cout << "\nQuery 0 predicted size=" << results.size() << ", truth="
                          << ground_truth[i].size() << std::endl;
                std::cout << "Predicted (first 10):";
                for (size_t j = 0; j < std::min<size_t>(10, results.size()); ++j) {
                    std::cout << " (" << results[j].first << "," << results[j].second << ")";
                }
                std::cout << "\nTruth (first 10):";
                for (size_t j = 0; j < std::min<size_t>(10, ground_truth[i].size()); ++j) {
                    std::cout << " (" << ground_truth[i][j].first << "," << ground_truth[i][j].second << ")";
                }
                std::cout << "\n" << std::endl;
            }

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
                {"nodes", stats.nodes},
                {"neighbor_bytes", stats.neighbor_bytes},
                {"total_bytes", stats.total_bytes()}
            });
        }

        benchmark_utils::json output = {
            {"strategy", "role_partition"},
            {"storage", {
                {"vector_bytes", vector_bytes},
                {"graph_bytes", total_graph_bytes},
                {"total_bytes", total_bytes},
                {"vector_mb", vector_bytes / 1024.0 / 1024.0},
                {"graph_mb", total_graph_bytes / 1024.0 / 1024.0},
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

        const auto output_path = root / "logical_partition_benchmark/benchmark/src/role_partition_results.json";
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
