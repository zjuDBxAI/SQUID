#include "pointer_hnsw_index.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace pointer_benchmark {

PointerHNSWIndex::PointerHNSWIndex(
    std::shared_ptr<SharedVectorTable> shared_table,
    int M,
    int efConstruction
) : shared_table_(shared_table),
    hnsw_index_(nullptr),
    M_(M),
    efConstruction_(efConstruction),
    ef_search_(100),
    force_copy_vectors_(false) {
}

void PointerHNSWIndex::build_from_partition(
    const std::string& conn_info,
    const std::string& partition_table_name
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Get all (document_id, block_id) from this partition
    std::string query = "SELECT document_id, block_id FROM " + partition_table_name +
                       " ORDER BY document_id, block_id";
    pqxx::result result = txn.exec(query);
    txn.commit();

    doc_block_ids_.clear();
    doc_block_ids_.reserve(result.size());

    // Collect indices into the shared table for this partition
    std::vector<faiss::idx_t> global_indices;
    global_indices.reserve(result.size());

    for (const auto& row : result) {
        int doc_id = row[0].as<int>();
        int block_id = row[1].as<int>();

        // Get the internal index in shared table
        faiss::idx_t global_idx = shared_table_->get_internal_index(doc_id, block_id);
        if (global_idx < 0) {
            std::cerr << "Warning: vector not found in shared table for doc_id="
                      << doc_id << ", block_id=" << block_id << std::endl;
            continue;
        }

        global_indices.push_back(global_idx);
        doc_block_ids_.push_back({doc_id, block_id});
    }

    // Create IndexHNSW with shared storage using the official Faiss API
    // Pass the shared IndexFlat as storage (non-owning pointer)
    hnsw_index_ = std::make_unique<faiss::IndexHNSW>(
        shared_table_->get_index_flat(),  // Shared storage
        M_                                 // HNSW M parameter
    );

    // Configure HNSW parameters
    hnsw_index_->hnsw.efConstruction = efConstruction_;
    hnsw_index_->own_fields = false;  // Don't delete the shared storage
    hnsw_index_->set_use_optimizations(false);

    if (force_copy_vectors_) {
        std::cerr << "[PointerHNSWIndex] force_copy_vectors_=true is not supported with"
                  << " shared storage. Use add_from_storage_ids instead." << std::endl;
    }

    // Build only the HNSW graph over the vectors already loaded in the shared table.
    if (!global_indices.empty()) {
        hnsw_index_->add_from_storage_ids(
            global_indices.size(),
            global_indices.data());
    }

    std::cout << "Built pointer HNSW index for " << partition_table_name
              << " with " << doc_block_ids_.size() << " vectors (shared storage)" << std::endl;
}

std::vector<PointerHNSWIndex::DocBlockId> PointerHNSWIndex::search(
    const float* query,
    int k
) {
    // Set efSearch
    hnsw_index_->hnsw.efSearch = ef_search_;

    // Perform search
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);

    hnsw_index_->search(1, query, k, distances.data(), indices.data());

    // Convert indices to (document_id, block_id)
    // HNSW assigns local, contiguous labels in the same order vectors were
    // added when we built this partition's graph. We stored that order in
    // doc_block_ids_, so we can translate the labels directly without
    // round-tripping through the shared table.
    std::vector<DocBlockId> results;
    results.reserve(k);

    for (int i = 0; i < k; i++) {
        faiss::idx_t label = indices[i];
        if (label < 0) {
            continue;
        }

        if (hnsw_index_->use_storage_ids &&
            label < static_cast<faiss::idx_t>(hnsw_index_->storage_ids.size())) {
            label = hnsw_index_->storage_ids[label];
        }

        DocBlockId mapped = shared_table_->get_doc_block_id(label);
        if (mapped.first == -1 && mapped.second == -1 &&
            label < static_cast<faiss::idx_t>(doc_block_ids_.size())) {
            mapped = doc_block_ids_[label];
        }

        if (mapped.first != -1 && mapped.second != -1) {
            results.push_back(mapped);
        }
    }

    return results;
}

void PointerHNSWIndex::save_graph(const std::string& path) const {
    if (!hnsw_index_) {
        return;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing pointer HNSW graph" << std::endl;
        return;
    }

    const faiss::HNSW& hnsw = hnsw_index_->hnsw;
    uint32_t magic = 0x574E5348; // 'HNSW'
    uint32_t version = 1;
    int64_t ntotal = hnsw_index_->ntotal;
    int32_t entry_point = hnsw.entry_point;
    int32_t max_level = hnsw.max_level;

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&ntotal), sizeof(ntotal));
    out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
    out.write(reinterpret_cast<const char*>(&max_level), sizeof(max_level));

    size_t storage_ids_size = hnsw_index_->storage_ids.size();
    out.write(reinterpret_cast<const char*>(&storage_ids_size), sizeof(storage_ids_size));
    if (storage_ids_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw_index_->storage_ids.data()),
                  storage_ids_size * sizeof(faiss::idx_t));
    }

    size_t offsets_size = hnsw.offsets.size();
    out.write(reinterpret_cast<const char*>(&offsets_size), sizeof(offsets_size));
    if (offsets_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw.offsets.data()), offsets_size * sizeof(size_t));
    }

    size_t neighbors_size = hnsw.neighbors.size();
    out.write(reinterpret_cast<const char*>(&neighbors_size), sizeof(neighbors_size));
    if (neighbors_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw.neighbors.data()),
                  neighbors_size * sizeof(faiss::HNSW::storage_idx_t));
    }

    size_t levels_size = hnsw.levels.size();
    out.write(reinterpret_cast<const char*>(&levels_size), sizeof(levels_size));
    if (levels_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw.levels.data()), levels_size * sizeof(int));
    }

    size_t assign_size = hnsw.assign_probas.size();
    out.write(reinterpret_cast<const char*>(&assign_size), sizeof(assign_size));
    if (assign_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw.assign_probas.data()), assign_size * sizeof(double));
    }

    size_t cum_size = hnsw.cum_nneighbor_per_level.size();
    out.write(reinterpret_cast<const char*>(&cum_size), sizeof(cum_size));
    if (cum_size > 0) {
        out.write(reinterpret_cast<const char*>(hnsw.cum_nneighbor_per_level.data()),
                  cum_size * sizeof(int));
    }

    size_t doc_map_size = doc_block_ids_.size();
    out.write(reinterpret_cast<const char*>(&doc_map_size), sizeof(doc_map_size));
    for (const auto& [doc, block] : doc_block_ids_) {
        out.write(reinterpret_cast<const char*>(&doc), sizeof(doc));
        out.write(reinterpret_cast<const char*>(&block), sizeof(block));
    }

    out.close();
}

bool PointerHNSWIndex::load_graph(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    bool has_header = (magic == 0x574E5348);
    uint32_t version = 0;
    int64_t ntotal = 0;
    int32_t entry_point = 0;
    int32_t max_level = 0;

    if (has_header) {
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&ntotal), sizeof(ntotal));
        in.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
        in.read(reinterpret_cast<char*>(&max_level), sizeof(max_level));
    } else {
        uint32_t high = 0;
        in.read(reinterpret_cast<char*>(&high), sizeof(high));
        ntotal = (static_cast<int64_t>(high) << 32) | magic;
        // entry_point/max_level will be computed from levels later
    }

    size_t storage_ids_size = 0;
    in.read(reinterpret_cast<char*>(&storage_ids_size), sizeof(storage_ids_size));
    std::vector<faiss::idx_t> storage_ids(storage_ids_size);
    if (storage_ids_size > 0) {
        in.read(reinterpret_cast<char*>(storage_ids.data()), storage_ids_size * sizeof(faiss::idx_t));
    }

    size_t offsets_size = 0;
    in.read(reinterpret_cast<char*>(&offsets_size), sizeof(offsets_size));
    std::vector<size_t> offsets(offsets_size);
    if (offsets_size > 0) {
        in.read(reinterpret_cast<char*>(offsets.data()), offsets_size * sizeof(size_t));
    }

    size_t neighbors_size = 0;
    in.read(reinterpret_cast<char*>(&neighbors_size), sizeof(neighbors_size));
    std::vector<faiss::HNSW::storage_idx_t> neighbors(neighbors_size);
    if (neighbors_size > 0) {
        in.read(reinterpret_cast<char*>(neighbors.data()), neighbors_size * sizeof(faiss::HNSW::storage_idx_t));
    }

    size_t levels_size = 0;
    in.read(reinterpret_cast<char*>(&levels_size), sizeof(levels_size));
    std::vector<int> levels(levels_size);
    if (levels_size > 0) {
        in.read(reinterpret_cast<char*>(levels.data()), levels_size * sizeof(int));
    }

    size_t assign_size = 0;
    in.read(reinterpret_cast<char*>(&assign_size), sizeof(assign_size));
    std::vector<double> assign(assign_size);
    if (assign_size > 0) {
        in.read(reinterpret_cast<char*>(assign.data()), assign_size * sizeof(double));
    }

    size_t cum_size = 0;
    in.read(reinterpret_cast<char*>(&cum_size), sizeof(cum_size));
    std::vector<int> cum(cum_size);
    if (cum_size > 0) {
        in.read(reinterpret_cast<char*>(cum.data()), cum_size * sizeof(int));
    }

    size_t doc_map_size = 0;
    in.read(reinterpret_cast<char*>(&doc_map_size), sizeof(doc_map_size));
    std::vector<DocBlockId> doc_map(doc_map_size);
    for (size_t i = 0; i < doc_map_size; ++i) {
        in.read(reinterpret_cast<char*>(&doc_map[i].first), sizeof(doc_map[i].first));
        in.read(reinterpret_cast<char*>(&doc_map[i].second), sizeof(doc_map[i].second));
    }

    if (!in) {
        std::cerr << "Failed to read pointer HNSW graph from " << path << std::endl;
        return false;
    }

    hnsw_index_ = std::make_unique<faiss::IndexHNSW>(shared_table_->get_index_flat(), M_);
    hnsw_index_->hnsw.efConstruction = efConstruction_;
    hnsw_index_->own_fields = false;
    hnsw_index_->set_use_optimizations(false);
    hnsw_index_->storage_ids = std::move(storage_ids);
    hnsw_index_->use_storage_ids = !hnsw_index_->storage_ids.empty();
    hnsw_index_->ntotal = ntotal;

    faiss::HNSW& hnsw = hnsw_index_->hnsw;
    hnsw.offsets = std::move(offsets);
    hnsw.levels = std::move(levels);
    hnsw.assign_probas = std::move(assign);
    hnsw.cum_nneighbor_per_level = std::move(cum);
    hnsw.neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>(std::move(neighbors));

    if (has_header) {
        hnsw.entry_point = entry_point;
        hnsw.max_level = max_level;
    } else {
        int best_level = -1;
        int best_node = 0;
        for (size_t i = 0; i < hnsw.levels.size(); ++i) {
            int level = hnsw.levels[i] - 1;
            if (level > best_level) {
                best_level = level;
                best_node = static_cast<int>(i);
            }
        }
        hnsw.max_level = best_level;
        hnsw.entry_point = best_node;
    }

    doc_block_ids_ = std::move(doc_map);

    return true;
}

} // namespace pointer_benchmark
