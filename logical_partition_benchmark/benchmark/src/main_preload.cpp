#include <algorithm>
#include <chrono>
#include <cstring>
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

#include "benchmark_utils.h"
#include "independent_hnsw_index.h"
#include "shared_vector_table.h"
#include <nlohmann/json.hpp>

using pointer_benchmark::IndependentHNSWIndex;
using pointer_benchmark::SharedVectorTable;
using benchmark_utils::DocBlockId;
using benchmark_utils::DocBlockIdHash;
using benchmark_utils::Query;

namespace {

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
    std::string save_index_dir;
};

Config load_config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file at " + config_path.string());
    }
    nlohmann::json j;
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
    if (j.contains("save_index_dir")) cfg.save_index_dir = j["save_index_dir"];

    if (j.contains("M")) cfg.M = j["M"].get<int>();
    if (j.contains("efConstruction")) cfg.efConstruction = j["efConstruction"].get<int>();
    if (j.contains("efSearch")) cfg.efSearch = j["efSearch"].get<int>();
    if (j.contains("topk")) cfg.topk = j["topk"].get<int>();

    return cfg;
}

std::string get_connection_string(const Config& cfg) {
    return "host=" + cfg.db_host +
           " port=" + cfg.db_port +
           " dbname=" + cfg.db_name +
           " user=" + cfg.db_user +
           " password=" + cfg.db_password;
}

constexpr size_t kAlignment = 64;

inline size_t align_up(size_t value, size_t alignment = kAlignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

class IndependentPartitionArena {
public:
    struct Entry {
        std::string name;
        std::unique_ptr<IndependentHNSWIndex> index;
        size_t vector_offset = 0;
        size_t neighbor_offset = 0;
        size_t vector_bytes = 0;
        size_t neighbor_bytes = 0;
        size_t neighbor_count = 0;
    };

    void preload(const std::vector<std::string>& partitions,
                 int dimension,
                 int M,
                 int efConstruction,
                 int efSearch,
                 const std::string& conn_info,
                 const std::filesystem::path& base_dir) {
        std::filesystem::create_directories(base_dir);

        entries_.clear();
        name_to_index_.clear();
        arena_.reset();
        total_vector_bytes_ = 0;
        total_neighbor_bytes_ = 0;

        std::vector<IndependentHNSWIndex::ColocatedSlice> slices;
        slices.reserve(partitions.size());

        // First pass: load indices and gather slice info.
        for (const auto& partition : partitions) {
            auto idx = std::make_unique<IndependentHNSWIndex>(dimension, M, efConstruction);
            idx->set_ef_search(efSearch);

            std::filesystem::path partition_dir = base_dir / partition;
            std::filesystem::create_directories(partition_dir);
            if (!idx->load_components(partition_dir.string())) {
                std::cout << "[Arena] Building independent index for " << partition << std::endl;
                idx->build_from_partition(conn_info, partition);
                idx->set_ef_search(efSearch);
                idx->save_components(partition_dir.string());
            }

            auto slice = idx->get_colocated_slice();
            IndependentPartitionArena::Entry entry;
            entry.name = partition;
            entry.vector_bytes = slice.vector_bytes;
            entry.neighbor_bytes = slice.neighbor_count * sizeof(faiss::HNSW::storage_idx_t);
            entry.neighbor_count = slice.neighbor_count;
            entry.index = std::move(idx);
            entries_.push_back(std::move(entry));
            slices.push_back(slice);
        }

        // Compute offsets with alignment.
        size_t current_offset = 0;
        for (auto& entry : entries_) {
            entry.vector_offset = align_up(current_offset);
            current_offset = entry.vector_offset + entry.vector_bytes;
            entry.neighbor_offset = align_up(current_offset);
            current_offset = entry.neighbor_offset + entry.neighbor_bytes;

            total_vector_bytes_ += entry.vector_bytes;
            total_neighbor_bytes_ += entry.neighbor_bytes;
        }

        arena_ = std::make_shared<std::vector<uint8_t>>(current_offset);

        // Second pass: copy data into arena and rebind indices.
        for (size_t i = 0; i < entries_.size(); ++i) {
            auto& entry = entries_[i];
            const auto& slice = slices[i];

            uint8_t* base = arena_->data();
            if (entry.vector_bytes > 0 && slice.vector_data) {
                std::memcpy(base + entry.vector_offset, slice.vector_data, entry.vector_bytes);
            }
            if (entry.neighbor_bytes > 0 && slice.neighbor_data) {
                std::memcpy(base + entry.neighbor_offset,
                            slice.neighbor_data,
                            entry.neighbor_bytes);
            }

            entry.index->rebind_colocated_slice(arena_,
                                                entry.vector_offset,
                                                entry.neighbor_offset,
                                                entry.neighbor_count);

            name_to_index_[entry.name] = i;
        }

        std::cout << "[Arena] Packed " << entries_.size()
                  << " partitions into contiguous arena (" << (arena_->size() / 1024.0 / 1024.0)
                  << " MB)" << std::endl;
    }

    IndependentHNSWIndex* get(const std::string& partition) const {
        auto it = name_to_index_.find(partition);
        if (it == name_to_index_.end()) {
            return nullptr;
        }
        return entries_[it->second].index.get();
    }

    size_t vector_bytes() const { return total_vector_bytes_; }
    size_t neighbor_bytes() const { return total_neighbor_bytes_; }
    size_t arena_bytes() const { return arena_ ? arena_->size() : 0; }

private:
    std::vector<Entry> entries_;
    std::unordered_map<std::string, size_t> name_to_index_;
    std::shared_ptr<std::vector<uint8_t>> arena_;
    size_t total_vector_bytes_ = 0;
    size_t total_neighbor_bytes_ = 0;
};

} // namespace

int main() {
    try {
        const auto project_root = benchmark_utils::project_root();
        Config cfg = load_config(project_root / "config.json");
        auto index_root = project_root / "logical_partition_benchmark/benchmark/results";
        std::filesystem::create_directories(index_root);

        if (cfg.save_index_dir.empty()) {
            cfg.save_index_dir = benchmark_utils::load_index_storage_path().string();
        }
        std::filesystem::path base_dir(cfg.save_index_dir);
        std::filesystem::create_directories(base_dir);

        const std::string conn_info = get_connection_string(cfg);

        // Load queries
        auto queries = benchmark_utils::load_queries(
            project_root / "basic_benchmark/query_dataset.json");
        if (queries.empty()) {
            std::cerr << "No queries found; aborting." << std::endl;
            return EXIT_FAILURE;
        }
        const int dimension = static_cast<int>(queries[0].query_vector.size());

        // Load shared vector table once.
        std::filesystem::path shared_dir = base_dir / "pointer_shared";
        std::filesystem::create_directories(shared_dir);
        std::filesystem::path shared_vectors_path = shared_dir / "shared_vectors.bin";
        auto shared_table = benchmark_utils::load_shared_table(
            dimension,
            shared_vectors_path,
            conn_info);
        std::cout << "Shared table loaded with " << shared_table->size() << " vectors" << std::endl;

        // Fetch metadata
        auto partitions = benchmark_utils::fetch_tables_with_prefix(conn_info, "documentblocks_role_");
        auto user_roles = benchmark_utils::load_user_roles(conn_info);
        auto doc_roles = benchmark_utils::load_document_roles(conn_info);

        // Load ground truth from cache if available, otherwise compute and cache.
        std::string gt_cache_file = (project_root / "pointer_benchmark/ground_truth_cache.json").string();
        std::vector<std::vector<DocBlockId>> ground_truth;

        auto save_ground_truth = [&](const auto& gt) {
            nlohmann::json gt_json = nlohmann::json::array();
            for (const auto& gt_query : gt) {
                nlohmann::json gt_query_json = nlohmann::json::array();
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
            std::cout << "Loading ground truth from cache..." << std::endl;
            nlohmann::json gt_json;
            gt_file >> gt_json;
            gt_file.close();

            for (const auto& gt_query : gt_json) {
                std::vector<DocBlockId> gt_results;
                for (const auto& result : gt_query) {
                    gt_results.push_back({result[0].get<int>(), result[1].get<int>()});
                }
                ground_truth.push_back(std::move(gt_results));
            }
            std::cout << "Loaded ground truth for " << ground_truth.size() << " queries" << std::endl;
        }

        if (ground_truth.size() != queries.size()) {
            if (!ground_truth.empty()) {
                std::cout << "Ground truth cache size mismatch (" << ground_truth.size()
                          << " vs " << queries.size() << "). Recomputing..." << std::endl;
            } else {
                std::cout << "Computing ground truth..." << std::endl;
            }

            ground_truth = benchmark_utils::compute_ground_truth(
                queries,
                shared_table,
                user_roles,
                doc_roles);

            std::cout << "Saving ground truth to cache..." << std::endl;
            save_ground_truth(ground_truth);
            std::cout << "Ground truth cached to " << gt_cache_file << std::endl;
        }

        // Preload independent partitions into contiguous arena.
        IndependentPartitionArena arena;
        std::filesystem::path independent_dir = base_dir / "independent";
        arena.preload(partitions,
                      dimension,
                      cfg.M,
                      cfg.efConstruction,
                      cfg.efSearch,
                      conn_info,
                      independent_dir);

        // Evaluate queries with preloaded independent indexes.
        std::vector<double> query_times;
        std::vector<double> recalls;
        query_times.reserve(queries.size());
        recalls.reserve(queries.size());

        for (size_t qi = 0; qi < queries.size(); ++qi) {
            const auto& query = queries[qi];
            auto user_it = user_roles.find(query.user_id);
            if (user_it == user_roles.end() || user_it->second.empty()) {
                query_times.push_back(0.0);
                recalls.push_back(0.0);
                continue;
            }

            std::vector<DocBlockId> aggregated;
            std::vector<IndependentHNSWIndex*> active_indices;
            active_indices.reserve(user_it->second.size());

            for (int role_id : user_it->second) {
                std::string partition = "documentblocks_role_" + std::to_string(role_id);
                IndependentHNSWIndex* index = arena.get(partition);
                if (!index) {
                    continue;
                }
                // Warm up to ensure timed path avoids cold-start penalties.
                index->search(query.query_vector.data(), cfg.topk);
                active_indices.push_back(index);
            }
            aggregated.reserve(static_cast<size_t>(cfg.topk) * active_indices.size());

            const auto start = std::chrono::high_resolution_clock::now();
            for (auto* index : active_indices) {
                auto results = index->search(query.query_vector.data(), cfg.topk);
                aggregated.insert(aggregated.end(), results.begin(), results.end());
            }
            const auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            query_times.push_back(elapsed_ms);

            const auto& gt = (qi < ground_truth.size()) ? ground_truth[qi]
                                                        : std::vector<DocBlockId>{};
            recalls.push_back(benchmark_utils::calculate_recall(aggregated, gt));
        }

        const double avg_time = query_times.empty()
                                    ? 0.0
                                    : std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                                          static_cast<double>(query_times.size());
        const double avg_recall = recalls.empty()
                                      ? 0.0
                                      : std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                                            static_cast<double>(recalls.size());

        auto times_copy = query_times;
        double p50 = times_copy.empty() ? 0.0 : benchmark_utils::percentile(times_copy, 0.5);
        double p90 = times_copy.empty() ? 0.0 : benchmark_utils::percentile(times_copy, 0.9);
        double p95 = times_copy.empty() ? 0.0 : benchmark_utils::percentile(times_copy, 0.95);
        double p99 = times_copy.empty() ? 0.0 : benchmark_utils::percentile(times_copy, 0.99);

        std::cout << "\n========= Independent HNSW (Preloaded Arena) =========" << std::endl;
        std::cout << "Arena bytes: " << (arena.arena_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  - Vector payload: " << (arena.vector_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  - Neighbor payload: " << (arena.neighbor_bytes() / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Latency:" << std::endl;
        std::cout << "  Avg: " << std::fixed << std::setprecision(3) << avg_time << " ms"
                  << "  P50: " << p50 << " ms"
                  << "  P90: " << p90 << " ms"
                  << "  P95: " << p95 << " ms"
                  << "  P99: " << p99 << " ms" << std::endl;
        std::cout << "Average recall: " << std::setprecision(4) << avg_recall << std::endl;

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
