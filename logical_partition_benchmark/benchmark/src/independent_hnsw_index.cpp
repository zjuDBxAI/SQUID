#include "independent_hnsw_index.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pqxx/pqxx>
#include <sstream>
#include <thread>

#include <faiss/utils/omp_utils.h>

namespace pointer_benchmark {
namespace {

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

constexpr const char* kIndexFilename = "integrated_index.bin";
constexpr const char* kMetadataFilename = "metadata.bin";

} // namespace

IndependentHNSWIndex::IndependentHNSWIndex(int dimension,
                                           int M,
                                           int efConstruction,
                                           faiss::MetricType metric,
                                           bool normalize)
        : dimension_(dimension),
          M_(M),
          efConstruction_(efConstruction),
          ef_search_(100),
          metric_(metric),
          normalize_(normalize),
          hnsw_index_(nullptr) {
}

IndependentHNSWIndex::ColocatedSlice IndependentHNSWIndex::get_colocated_slice() const {
    return {};
}

void IndependentHNSWIndex::rebind_colocated_slice(
        const std::shared_ptr<std::vector<uint8_t>>& /*buffer*/,
        size_t /*vector_offset_bytes*/,
        size_t /*neighbor_offset_bytes*/,
        size_t /*neighbor_count*/) {
    // No-op for integrated storage: vectors already colocated with nodes.
}

void IndependentHNSWIndex::build_from_partition(
        const std::string& conn_info,
        const std::string& partition_table_name) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    std::string query = "SELECT document_id, block_id, vector FROM " + partition_table_name +
            " ORDER BY document_id, block_id";
    pqxx::result result = txn.exec(query);
    txn.commit();

    doc_block_ids_.clear();
    doc_block_ids_.reserve(result.size());

    std::vector<float> all_vectors_flat;
    all_vectors_flat.reserve(result.size() * static_cast<size_t>(dimension_));

    for (const auto& row : result) {
        int doc_id = row[0].as<int>();
        int block_id = row[1].as<int>();
        std::string vector_str = row[2].as<std::string>();

        std::vector<float> vector;
        vector.reserve(dimension_);

        if (!vector_str.empty() && vector_str.front() == '[') {
            vector_str = vector_str.substr(1);
        }
        if (!vector_str.empty() && vector_str.back() == ']') {
            vector_str.pop_back();
        }

        std::stringstream ss(vector_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            vector.push_back(std::stof(item));
        }

        if (vector.size() != static_cast<size_t>(dimension_)) {
            std::cerr << "Warning: vector dimension mismatch for doc_id="
                      << doc_id << ", block_id=" << block_id << std::endl;
            continue;
        }

        if (normalize_) {
            normalize_vector(vector);
        }

        doc_block_ids_.push_back({doc_id, block_id});
        all_vectors_flat.insert(all_vectors_flat.end(), vector.begin(), vector.end());
    }

    hnsw_index_ = std::make_unique<faiss::IndexHNSWLib>(
            dimension_, M_, metric_, efConstruction_, ef_search_);
    hnsw_index_->set_efSearch(ef_search_);

    if (!all_vectors_flat.empty()) {
        hnsw_index_->add(static_cast<faiss::idx_t>(doc_block_ids_.size()), all_vectors_flat.data());
    }

    std::cout << "Built integrated HNSW index for " << partition_table_name
              << " with " << doc_block_ids_.size() << " vectors (integrated storage)" << std::endl;
}

std::vector<IndependentHNSWIndex::DocBlockId> IndependentHNSWIndex::search(
        const float* query,
        int k) {
    if (!hnsw_index_) {
        return {};
    }

    hnsw_index_->set_efSearch(ef_search_);

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);

    hnsw_index_->search(1, query, k, distances.data(), indices.data());

    std::vector<DocBlockId> results;
    results.reserve(k);

    for (int i = 0; i < k; i++) {
        if (indices[i] >= 0 && indices[i] < static_cast<faiss::idx_t>(doc_block_ids_.size())) {
            results.push_back(doc_block_ids_[indices[i]]);
        }
    }

    return results;
}

size_t IndependentHNSWIndex::memory_bytes() const {
    if (!hnsw_index_) {
        return 0;
    }
    return hnsw_index_->allocated_bytes();
}

void IndependentHNSWIndex::save_components(const std::string& base_dir) const {
    if (!hnsw_index_) {
        return;
    }

    std::filesystem::path dir(base_dir);
    std::filesystem::create_directories(dir);

    std::filesystem::path index_path = dir / kIndexFilename;
    if (!hnsw_index_->save(index_path.string())) {
        std::cerr << "Failed to save integrated HNSW index to " << index_path << std::endl;
    }

    std::filesystem::path meta_path = dir / kMetadataFilename;
    std::ofstream meta(meta_path, std::ios::binary);
    if (!meta) {
        std::cerr << "Failed to open " << meta_path << " for writing metadata" << std::endl;
        return;
    }

    int32_t version = 1;
    int32_t dim = dimension_;
    int32_t m = M_;
    int32_t efc = efConstruction_;
    int32_t metric = static_cast<int32_t>(metric_);
    uint8_t normalize_flag = normalize_ ? 1 : 0;
    uint64_t count = doc_block_ids_.size();

    meta.write(reinterpret_cast<const char*>(&version), sizeof(version));
    meta.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    meta.write(reinterpret_cast<const char*>(&m), sizeof(m));
    meta.write(reinterpret_cast<const char*>(&efc), sizeof(efc));
    meta.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    meta.write(reinterpret_cast<const char*>(&normalize_flag), sizeof(normalize_flag));
    meta.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [doc, block] : doc_block_ids_) {
        meta.write(reinterpret_cast<const char*>(&doc), sizeof(doc));
        meta.write(reinterpret_cast<const char*>(&block), sizeof(block));
    }
}

bool IndependentHNSWIndex::load_components(const std::string& base_dir) {
    std::filesystem::path dir(base_dir);
    std::filesystem::path index_path = dir / kIndexFilename;
    std::filesystem::path meta_path = dir / kMetadataFilename;

    if (!std::filesystem::exists(index_path) || !std::filesystem::exists(meta_path)) {
        return false;
    }

    std::ifstream meta(meta_path, std::ios::binary);
    if (!meta) {
        std::cerr << "Failed to open " << meta_path << " for reading metadata" << std::endl;
        return false;
    }

    int32_t version = 0;
    int32_t dim = 0;
    int32_t m = 0;
    int32_t efc = 0;
    int32_t metric = 0;
    uint8_t normalize_flag = 0;
    uint64_t count = 0;

    meta.read(reinterpret_cast<char*>(&version), sizeof(version));
    meta.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    meta.read(reinterpret_cast<char*>(&m), sizeof(m));
    meta.read(reinterpret_cast<char*>(&efc), sizeof(efc));
    meta.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    meta.read(reinterpret_cast<char*>(&normalize_flag), sizeof(normalize_flag));
    meta.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (dim != dimension_ || m != M_) {
        std::cerr << "Metadata mismatch when loading integrated index from " << base_dir << std::endl;
        return false;
    }

    doc_block_ids_.resize(count);
    for (uint64_t i = 0; i < count; ++i) {
        int doc = 0;
        int block = 0;
        meta.read(reinterpret_cast<char*>(&doc), sizeof(doc));
        meta.read(reinterpret_cast<char*>(&block), sizeof(block));
        doc_block_ids_[i] = {doc, block};
    }

    hnsw_index_ = std::make_unique<faiss::IndexHNSWLib>(
            dimension_, M_, static_cast<faiss::MetricType>(metric), efConstruction_, ef_search_);
    if (!hnsw_index_->load(index_path.string())) {
        std::cerr << "Failed to load integrated HNSW index from " << index_path << std::endl;
        hnsw_index_.reset();
        doc_block_ids_.clear();
        return false;
    }
    hnsw_index_->set_efSearch(ef_search_);

    normalize_ = normalize_flag != 0;

    return true;
}

} // namespace pointer_benchmark
#include <omp.h>
