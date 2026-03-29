#include "shared_vector_table.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <limits>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace pointer_benchmark {

// Removed custom ScalarIndexFlat implementation in favor of Faiss IndexFlat.

namespace {

std::unique_ptr<faiss::IndexFlat> make_index(int dimension, faiss::MetricType metric) {
    return std::make_unique<faiss::IndexFlat>(dimension, metric);
}

void normalize_vector(std::vector<float>& vec) {
    float norm_sq = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f);
    if (norm_sq <= 0.0f) {
        return;
    }
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& value : vec) {
        value *= inv_norm;
    }
}

void normalize_buffer(float* buffer, faiss::idx_t count, int dimension) {
    if (!buffer) {
        return;
    }
    for (faiss::idx_t i = 0; i < count; ++i) {
        float* vec = buffer + (static_cast<size_t>(i) * static_cast<size_t>(dimension));
        float norm_sq = 0.0f;
        for (int d = 0; d < dimension; ++d) {
            norm_sq += vec[d] * vec[d];
        }
        if (norm_sq <= 0.0f) {
            continue;
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (int d = 0; d < dimension; ++d) {
            vec[d] *= inv_norm;
        }
    }
}

} // namespace

SharedVectorTable::SharedVectorTable(int dimension,
                                     faiss::MetricType metric,
                                     bool normalize)
    : dimension_(dimension),
      metric_(metric),
      normalize_(normalize),
      index_flat_(make_index(dimension_, metric_)) {
}

size_t SharedVectorTable::load_from_database(const std::string& conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    pqxx::result result = txn.exec(
        "SELECT document_id, block_id, vector FROM documentblocks "
        "ORDER BY document_id, block_id"
    );

    id_to_index_.clear();
    index_to_id_.clear();
    std::vector<float> all_vectors_flat;
    all_vectors_flat.reserve(result.size() * static_cast<size_t>(dimension_));

    for (const auto& row : result) {
        int doc_id = row[0].as<int>();
        int block_id = row[1].as<int>();
        std::string vector_str = row[2].as<std::string>();

        std::vector<float> vector;
        vector.reserve(dimension_);

        if (!vector_str.empty() && vector_str.front() == '[') vector_str = vector_str.substr(1);
        if (!vector_str.empty() && vector_str.back() == ']') vector_str.pop_back();

        std::stringstream ss(vector_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            vector.push_back(std::stof(item));
        }

        if (vector.size() != static_cast<size_t>(dimension_)) {
            std::cerr << "Warning: vector dimension mismatch for doc_id=" << doc_id
                      << ", block_id=" << block_id << std::endl;
            continue;
        }

        if (normalize_) {
            normalize_vector(vector);
        }

        DocBlockId id = {doc_id, block_id};
        faiss::idx_t internal_index = static_cast<faiss::idx_t>(index_to_id_.size());
        id_to_index_[id] = internal_index;
        index_to_id_.push_back(id);

        all_vectors_flat.insert(all_vectors_flat.end(), vector.begin(), vector.end());
    }

    txn.commit();

    index_flat_ = make_index(dimension_, metric_);
    const faiss::idx_t count = static_cast<faiss::idx_t>(index_to_id_.size());
    if (count > 0) {
        index_flat_->add(count, all_vectors_flat.data());
    }

    std::cout << "Loaded " << index_to_id_.size() << " vectors into shared vector table" << std::endl;
    return index_to_id_.size();
}

void SharedVectorTable::flush_cache() const {
    if (index_flat_) {
        // No explicit cache flush when using standard Faiss IndexFlat.
    }
}

std::unique_ptr<SharedVectorTable> SharedVectorTable::clone() const {
    auto clone_table = std::make_unique<SharedVectorTable>(dimension_, metric_, normalize_);
    if (index_flat_ && index_flat_->ntotal > 0) {
        clone_table->index_flat_->add(index_flat_->ntotal, index_flat_->get_xb());
    }
    clone_table->id_to_index_ = id_to_index_;
    clone_table->index_to_id_ = index_to_id_;
    return clone_table;
}


const float* SharedVectorTable::get_vector(int doc_id, int block_id) const {
    auto it = id_to_index_.find({doc_id, block_id});
    if (it == id_to_index_.end()) {
        return nullptr;
    }
    return index_flat_->get_xb() + static_cast<size_t>(it->second) * static_cast<size_t>(dimension_);
}

faiss::idx_t SharedVectorTable::get_internal_index(int doc_id, int block_id) const {
    auto it = id_to_index_.find({doc_id, block_id});
    if (it == id_to_index_.end()) {
        return -1;
    }
    return it->second;
}

SharedVectorTable::DocBlockId SharedVectorTable::get_doc_block_id(faiss::idx_t internal_idx) const {
    if (internal_idx < 0 || internal_idx >= static_cast<faiss::idx_t>(index_to_id_.size())) {
        return {-1, -1};
    }
    return index_to_id_[static_cast<size_t>(internal_idx)];
}

void SharedVectorTable::save_vectors(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing shared vectors" << std::endl;
        return;
    }

    int32_t dim = dimension_;
    faiss::idx_t count = index_flat_->ntotal;
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    if (count > 0) {
        size_t total_floats = static_cast<size_t>(count) * static_cast<size_t>(dimension_);
        const float* xb = index_flat_->get_xb();
        out.write(reinterpret_cast<const char*>(xb), total_floats * sizeof(float));
    }

    out.close();

    std::ofstream meta(path + ".meta", std::ios::binary);
    if (!meta) {
        std::cerr << "Failed to open " << path << ".meta for writing metadata" << std::endl;
        return;
    }

    meta.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    meta.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& id : index_to_id_) {
        meta.write(reinterpret_cast<const char*>(&id.first), sizeof(id.first));
        meta.write(reinterpret_cast<const char*>(&id.second), sizeof(id.second));
    }
}

bool SharedVectorTable::load_vectors(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    int32_t dim = 0;
    faiss::idx_t count = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) {
        std::cerr << "Malformed shared vector file: " << path << std::endl;
        return false;
    }

    std::vector<float> buffer(static_cast<size_t>(count) * static_cast<size_t>(dim));
    if (!buffer.empty()) {
        in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
        if (!in) {
            std::cerr << "Failed to read vector payload from " << path << std::endl;
            return false;
        }
    }
    in.close();

    std::ifstream meta(path + ".meta", std::ios::binary);
    if (!meta) {
        std::cerr << "Missing shared vector metadata file: " << path << ".meta" << std::endl;
        return false;
    }

    int32_t meta_dim = 0;
    faiss::idx_t meta_count = 0;
    meta.read(reinterpret_cast<char*>(&meta_dim), sizeof(meta_dim));
    meta.read(reinterpret_cast<char*>(&meta_count), sizeof(meta_count));
    if (!meta || meta_dim != dim || meta_count != count) {
        std::cerr << "Shared vector metadata mismatch for " << path << std::endl;
        return false;
    }

    std::vector<DocBlockId> doc_map(static_cast<size_t>(meta_count));
    for (size_t i = 0; i < doc_map.size(); ++i) {
        meta.read(reinterpret_cast<char*>(&doc_map[i].first), sizeof(doc_map[i].first));
        meta.read(reinterpret_cast<char*>(&doc_map[i].second), sizeof(doc_map[i].second));
        if (!meta) {
            std::cerr << "Failed to read shared vector metadata entries" << std::endl;
            return false;
        }
    }

    dimension_ = dim;
    index_flat_ = make_index(dimension_, metric_);
    if (normalize_ && count > 0) {
        normalize_buffer(buffer.data(), count, dimension_);
    }
    if (count > 0) {
        index_flat_->add(count, buffer.data());
    }

    id_to_index_.clear();
    index_to_id_.clear();
    index_to_id_.reserve(doc_map.size());
    for (size_t i = 0; i < doc_map.size(); ++i) {
        index_to_id_.push_back(doc_map[i]);
        id_to_index_[doc_map[i]] = static_cast<faiss::idx_t>(i);
    }

    return true;
}

} // namespace pointer_benchmark
