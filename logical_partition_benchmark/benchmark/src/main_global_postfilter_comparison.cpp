#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "benchmark_utils.h"
#include "global_hnsw_index.h"
#include "shared_vector_table.h"

using pointer_benchmark::GlobalHNSWIndex;
using pointer_benchmark::IndexMode;
using pointer_benchmark::SharedVectorTable;
using benchmark_utils::json;

namespace {

struct GlobalMetrics {
    std::string mode_name;
    size_t vector_bytes = 0;
    size_t graph_bytes = 0;
    double avg_time = 0.0;
    double avg_recall = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    size_t nodes = 0;

    size_t total_bytes() const { return vector_bytes + graph_bytes; }
};

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

void evaluate_global_postfilter(
    const std::string& mode_name,
    IndexMode mode,
    const std::filesystem::path& graph_path,
    int M,
    int efConstruction,
    int efSearch,
    const std::shared_ptr<SharedVectorTable>& shared_table,
    const std::vector<benchmark_utils::Query>& queries,
    const std::unordered_map<int, std::vector<faiss::idx_t>>& role_to_internal_ids,
    const std::unordered_map<int, std::unordered_set<int>>& user_roles,
    const std::vector<std::vector<SharedVectorTable::DocBlockId>>& ground_truth,
    GlobalMetrics& metrics_out,
    int dimension) {

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test: " << mode_name << std::endl;
    std::cout << "========================================" << std::endl;

    // Build or load index
    GlobalHNSWIndex global_index(mode, shared_table, M, efConstruction);
    if (!std::filesystem::exists(graph_path) || !global_index.load_graph(graph_path.string())) {
        std::cout << "Building global HNSW graph..." << std::endl;
        global_index.build();
        global_index.save_graph(graph_path.string());
    } else {
        std::cout << "Loaded global graph from disk" << std::endl;
    }
    global_index.set_ef_search(efSearch);

    // Compute storage stats
    const size_t ntotal = global_index.ntotal();
    if (mode == IndexMode::SHARED_POINTER) {
        metrics_out.vector_bytes = shared_table->get_index_flat()->ntotal * dimension * sizeof(float);
        const auto graph_stats = benchmark_utils::compute_hnsw_graph_stats(global_index.get_index()->hnsw);
        metrics_out.graph_bytes = graph_stats.total_bytes();
        metrics_out.nodes = graph_stats.nodes;
    } else {
        metrics_out.vector_bytes = ntotal * dimension * sizeof(float);
        metrics_out.graph_bytes = 0;  // Included in vector_bytes for INDEPENDENT mode
        metrics_out.nodes = ntotal;
    }

    std::cout << "\n--- Storage ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total storage: " << (metrics_out.total_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Vectors: " << (metrics_out.vector_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Graph: " << (metrics_out.graph_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Nodes: " << metrics_out.nodes << std::endl;

    // Query performance evaluation
    std::vector<double> query_times;
    std::vector<double> recalls;
    query_times.reserve(queries.size());
    recalls.reserve(queries.size());

    std::vector<float> temp_dist;
    std::vector<faiss::idx_t> temp_idx;
    const auto ntotal_val = (mode == IndexMode::SHARED_POINTER)
        ? shared_table->get_index_flat()->ntotal
        : ntotal;
    std::vector<uint8_t> mark(static_cast<size_t>(ntotal_val), 0);
    std::vector<faiss::idx_t> allowed_ids;
    std::vector<faiss::idx_t> touched_ids;

    std::cout << "\n--- Query Performance ---" << std::endl;
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto& query = queries[i];
        const auto roles_it = user_roles.find(query.user_id);

        std::vector<SharedVectorTable::DocBlockId> results;
        allowed_ids.clear();
        touched_ids.clear();

        if (roles_it != user_roles.end()) {
            for (int role_id : roles_it->second) {
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
        }

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

        // Clear mark
        for (faiss::idx_t id : touched_ids) {
            size_t pos = static_cast<size_t>(id);
            if (pos < mark.size()) {
                mark[pos] = 0;
            }
        }

        query_times.push_back(time_ms);
        recalls.push_back(benchmark_utils::calculate_recall(results, ground_truth[i]));

        if ((i + 1) % 100 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << queries.size() << std::endl;
        }
    }

    metrics_out.avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                           static_cast<double>(query_times.size());
    metrics_out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                             static_cast<double>(recalls.size());

    auto times_copy = query_times;
    metrics_out.p50 = benchmark_utils::percentile(times_copy, 0.5);
    metrics_out.p90 = benchmark_utils::percentile(times_copy, 0.9);
    metrics_out.p95 = benchmark_utils::percentile(times_copy, 0.95);
    metrics_out.p99 = benchmark_utils::percentile(times_copy, 0.99);

    std::cout << std::setprecision(2);
    std::cout << "Avg time: " << metrics_out.avg_time << " ms" << std::endl;
    std::cout << "P50: " << metrics_out.p50 << " ms, P90: " << metrics_out.p90
              << " ms, P95: " << metrics_out.p95 << " ms, P99: " << metrics_out.p99 << " ms"
              << std::endl;
    std::cout << "Avg recall: " << metrics_out.avg_recall << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Global HNSW Postfilter Comparison" << std::endl;
        std::cout << "Logical (Shared Storage) vs Physical (Independent Storage)" << std::endl;
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
        std::cout << "ef_search: " << globalEfSearch << std::endl;

        const std::filesystem::path base_dir = benchmark_utils::load_index_storage_path();
        std::cout << "Index storage path: " << base_dir << std::endl;

        // Setup directories
        const auto logical_dir = base_dir / "postfilter";
        const auto physical_dir = base_dir / "physical_postfilter";
        std::filesystem::create_directories(logical_dir);
        std::filesystem::create_directories(physical_dir);

        const auto shared_cache = logical_dir / "shared_vectors.bin";
        const auto logical_graph_path = logical_dir / "global_graph.bin";
        const auto physical_graph_path = physical_dir / "global_graph.bin";

        // Load shared table
        auto shared_table = benchmark_utils::load_shared_table(dimension, shared_cache, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        // Load roles and ground truth
        auto user_roles = benchmark_utils::load_user_roles(conn_info);
        auto doc_roles = benchmark_utils::load_document_roles(conn_info);
        auto doc_block_roles = benchmark_utils::load_doc_block_roles(conn_info);
        auto ground_truth = benchmark_utils::compute_ground_truth(queries, shared_table, user_roles, doc_roles);

        // Build role to internal IDs mapping
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

        // Evaluate both modes
        GlobalMetrics logical_metrics;
        logical_metrics.mode_name = "Logical (Shared Pointer)";
        evaluate_global_postfilter(
            "Global HNSW Postfilter - Logical (Shared Storage)",
            IndexMode::SHARED_POINTER,
            logical_graph_path,
            M, efConstruction, globalEfSearch,
            shared_table, queries, role_to_internal_ids, user_roles, ground_truth,
            logical_metrics, dimension);

        GlobalMetrics physical_metrics;
        physical_metrics.mode_name = "Physical (Independent)";
        evaluate_global_postfilter(
            "Global HNSW Postfilter - Physical (Independent Storage)",
            IndexMode::INDEPENDENT,
            physical_graph_path,
            M, efConstruction, globalEfSearch,
            shared_table, queries, role_to_internal_ids, user_roles, ground_truth,
            physical_metrics, dimension);

        // ===================================================================
        // COMPARISON
        // ===================================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "COMPARISON: Logical vs Physical Postfilter" << std::endl;
        std::cout << "========================================" << std::endl;

        double logical_total_mb = logical_metrics.total_bytes() / 1024.0 / 1024.0;
        double logical_vector_mb = logical_metrics.vector_bytes / 1024.0 / 1024.0;
        double logical_graph_mb = logical_metrics.graph_bytes / 1024.0 / 1024.0;
        double physical_total_mb = physical_metrics.total_bytes() / 1024.0 / 1024.0;
        double physical_vector_mb = physical_metrics.vector_bytes / 1024.0 / 1024.0;
        double physical_graph_mb = physical_metrics.graph_bytes / 1024.0 / 1024.0;

        double mem_savings_mb = physical_total_mb - logical_total_mb;
        double mem_savings_pct =
            (1.0 - static_cast<double>(logical_metrics.total_bytes()) /
                           static_cast<double>(physical_metrics.total_bytes())) * 100.0;

        double recall_diff = std::abs(logical_metrics.avg_recall - physical_metrics.avg_recall);
        double speed_ratio = logical_metrics.avg_time / physical_metrics.avg_time;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nMemory:" << std::endl;
        std::cout << "  Logical (shared):   " << logical_total_mb << " MB (vectors "
                  << logical_vector_mb << " MB, graph " << logical_graph_mb << " MB)" << std::endl;
        std::cout << "  Physical (independent): " << physical_total_mb << " MB (vectors "
                  << physical_vector_mb << " MB, graph " << physical_graph_mb << " MB)" << std::endl;
        std::cout << "  Logical savings:    " << mem_savings_mb << " MB (" << mem_savings_pct << "%)" << std::endl;

        std::cout << std::setprecision(4);
        std::cout << "\nRecall (avg):" << std::endl;
        std::cout << "  Logical:    " << logical_metrics.avg_recall << std::endl;
        std::cout << "  Physical:   " << physical_metrics.avg_recall << std::endl;
        std::cout << "  Difference: " << recall_diff << std::endl;

        std::cout << std::setprecision(2);
        std::cout << "\nSearch Speed (avg):" << std::endl;
        std::cout << "  Logical:    " << logical_metrics.avg_time << " ms/query" << std::endl;
        std::cout << "  Physical:   " << physical_metrics.avg_time << " ms/query" << std::endl;
        std::cout << "  Speed ratio: " << speed_ratio << "x" << std::endl;

        std::cout << "\nSearch Speed (percentiles):" << std::endl;
        std::cout << "  P50: Logical=" << logical_metrics.p50 << "ms, Physical=" << physical_metrics.p50 << "ms" << std::endl;
        std::cout << "  P90: Logical=" << logical_metrics.p90 << "ms, Physical=" << physical_metrics.p90 << "ms" << std::endl;
        std::cout << "  P95: Logical=" << logical_metrics.p95 << "ms, Physical=" << physical_metrics.p95 << "ms" << std::endl;
        std::cout << "  P99: Logical=" << logical_metrics.p99 << "ms, Physical=" << physical_metrics.p99 << "ms" << std::endl;

        // Save results
        json output = {
            {"config", {
                {"M", M},
                {"efConstruction", efConstruction},
                {"efSearch", globalEfSearch},
                {"num_queries", queries.size()}
            }},
            {"logical_postfilter", {
                {"total_memory_mb", logical_total_mb},
                {"vector_memory_mb", logical_vector_mb},
                {"graph_memory_mb", logical_graph_mb},
                {"nodes", logical_metrics.nodes},
                {"avg_time_ms", logical_metrics.avg_time},
                {"p50_ms", logical_metrics.p50},
                {"p90_ms", logical_metrics.p90},
                {"p95_ms", logical_metrics.p95},
                {"p99_ms", logical_metrics.p99},
                {"avg_recall", logical_metrics.avg_recall}
            }},
            {"physical_postfilter", {
                {"total_memory_mb", physical_total_mb},
                {"vector_memory_mb", physical_vector_mb},
                {"graph_memory_mb", physical_graph_mb},
                {"nodes", physical_metrics.nodes},
                {"avg_time_ms", physical_metrics.avg_time},
                {"p50_ms", physical_metrics.p50},
                {"p90_ms", physical_metrics.p90},
                {"p95_ms", physical_metrics.p95},
                {"p99_ms", physical_metrics.p99},
                {"avg_recall", physical_metrics.avg_recall}
            }},
            {"comparison", {
                {"memory_savings_mb", mem_savings_mb},
                {"memory_savings_percent", mem_savings_pct},
                {"speed_ratio", speed_ratio},
                {"recall_diff", recall_diff}
            }}
        };

        const auto output_path = root / "logical_partition_benchmark/benchmark/results/global_postfilter_comparison.json";
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
