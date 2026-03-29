#include "independent_hnsw_index.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pqxx/pqxx>
#include <sstream>
#include <faiss/IndexFlatCodes.h>

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

void normalize_buffer(float* buffer, faiss::idx_t count, int dimension) {
    if (!buffer) return;
    for (faiss::idx_t i = 0; i < count; ++i) {
        float* vec = buffer + static_cast<size_t>(i) * static_cast<size_t>(dimension);
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

struct ColocatedBufferOwner : faiss::MaybeOwnedVectorOwner {
    explicit ColocatedBufferOwner(std::shared_ptr<std::vector<uint8_t>> buffer)
        : buffer_(std::move(buffer)) {}
    std::shared_ptr<std::vector<uint8_t>> buffer_;
};

IndependentHNSWIndex::IndependentHNSWIndex(int dimension,
                                           int M,
                                           int efConstruction,
                                           faiss::MetricType metric,
                                           bool normalize,
                                           bool colocate_storage)
    : dimension_(dimension),
      M_(M),
      efConstruction_(efConstruction),
      ef_search_(100),
      metric_(metric),
      normalize_(normalize),
      colocate_storage_(colocate_storage),
      hnsw_index_(nullptr) {
}

void IndependentHNSWIndex::apply_colocated_layout() {
    if (!colocate_storage_ || !hnsw_index_) {
        return;
    }

    faiss::IndexFlatCodes* flat_codes =
        dynamic_cast<faiss::IndexFlatCodes*>(hnsw_index_->storage);
    if (!flat_codes) {
        std::cerr << "IndependentHNSWIndex: storage is not IndexFlatCodes, "
                     "skipping colocated layout."
                  << std::endl;
        return;
    }

    const size_t vector_bytes =
        static_cast<size_t>(hnsw_index_->ntotal) * static_cast<size_t>(dimension_) *
        sizeof(float);
    faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>& neighbors =
        hnsw_index_->hnsw.neighbors;
    const size_t neighbor_bytes =
        neighbors.size() * sizeof(faiss::HNSW::storage_idx_t);
    const size_t total_bytes = vector_bytes + neighbor_bytes;

    if (total_bytes == 0) {
        colocated_buffer_.reset();
        colocated_owner_.reset();
        return;
    }

    auto buffer = std::make_shared<std::vector<uint8_t>>(total_bytes);
    uint8_t* base_ptr = buffer->data();

    if (vector_bytes > 0 && flat_codes->codes.data()) {
        std::memcpy(base_ptr, flat_codes->codes.data(), vector_bytes);
    }
    if (neighbor_bytes > 0 && neighbors.data()) {
        std::memcpy(base_ptr + vector_bytes, neighbors.data(), neighbor_bytes);
    }

    auto owner = std::make_shared<ColocatedBufferOwner>(buffer);

    if (vector_bytes > 0) {
        flat_codes->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
            base_ptr, vector_bytes, owner);
    } else {
        flat_codes->codes = faiss::MaybeOwnedVector<uint8_t>();
    }

    if (neighbor_bytes > 0) {
        auto neighbor_ptr =
            reinterpret_cast<faiss::HNSW::storage_idx_t*>(base_ptr + vector_bytes);
        neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>::create_view(
            neighbor_ptr, neighbors.size(), owner);
    } else {
        neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>();
    }

    colocated_buffer_ = std::move(buffer);
    colocated_owner_ = std::move(owner);
}

IndependentHNSWIndex::ColocatedSlice IndependentHNSWIndex::get_colocated_slice() const {
    ColocatedSlice slice;
    if (!hnsw_index_) {
        return slice;
    }

    const auto* flat_codes = dynamic_cast<const faiss::IndexFlatCodes*>(hnsw_index_->storage);
    if (flat_codes) {
        slice.vector_data = flat_codes->codes.data();
        slice.vector_bytes = flat_codes->codes.size();
    }

    const auto& neighbors = hnsw_index_->hnsw.neighbors;
    slice.neighbor_data = neighbors.data();
    slice.neighbor_count = neighbors.size();
    return slice;
}

void IndependentHNSWIndex::rebind_colocated_slice(
    const std::shared_ptr<std::vector<uint8_t>>& buffer,
    size_t vector_offset_bytes,
    size_t neighbor_offset_bytes,
    size_t neighbor_count
) {
    if (!buffer || !hnsw_index_) {
        return;
    }

    auto* flat_codes = dynamic_cast<faiss::IndexFlatCodes*>(hnsw_index_->storage);
    if (!flat_codes) {
        std::cerr << "IndependentHNSWIndex: storage is not IndexFlatCodes, cannot rebind external buffer."
                  << std::endl;
        return;
    }

    const size_t vector_bytes = flat_codes->codes.size();
    const size_t neighbor_bytes = neighbor_count * sizeof(faiss::HNSW::storage_idx_t);

    if ((vector_bytes > 0 && vector_offset_bytes + vector_bytes > buffer->size()) ||
        (neighbor_bytes > 0 && neighbor_offset_bytes + neighbor_bytes > buffer->size())) {
        std::cerr << "IndependentHNSWIndex: external buffer too small for rebind request." << std::endl;
        return;
    }

    auto owner = std::make_shared<ColocatedBufferOwner>(buffer);
    uint8_t* base = buffer->data();

    if (vector_bytes > 0) {
        flat_codes->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
            base + vector_offset_bytes,
            vector_bytes,
            owner);
    } else {
        flat_codes->codes = faiss::MaybeOwnedVector<uint8_t>();
    }

    if (neighbor_count > 0) {
        auto neighbor_ptr = reinterpret_cast<faiss::HNSW::storage_idx_t*>(base + neighbor_offset_bytes);
        hnsw_index_->hnsw.neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>::create_view(
            neighbor_ptr,
            neighbor_count,
            owner);
    } else {
        hnsw_index_->hnsw.neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>();
    }

    colocated_buffer_ = buffer;
    colocated_owner_ = std::move(owner);
}

void IndependentHNSWIndex::build_from_partition(
    const std::string& conn_info,
    const std::string& partition_table_name
) {
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

        if (!vector_str.empty() && vector_str.front() == '[') vector_str = vector_str.substr(1);
        if (!vector_str.empty() && vector_str.back() == ']') vector_str.pop_back();

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

    hnsw_index_ = std::make_unique<faiss::IndexHNSWFlat>(dimension_, M_, metric_);
    hnsw_index_->hnsw.efConstruction = efConstruction_;

    if (!all_vectors_flat.empty()) {
        hnsw_index_->add(static_cast<faiss::idx_t>(doc_block_ids_.size()), all_vectors_flat.data());
    }

    apply_colocated_layout();

    std::cout << "Built independent HNSW index for " << partition_table_name
              << " with " << doc_block_ids_.size() << " vectors (own storage)" << std::endl;
}

std::vector<IndependentHNSWIndex::DocBlockId> IndependentHNSWIndex::search(
    const float* query,
    int k
) {
    hnsw_index_->hnsw.efSearch = ef_search_;

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
    if (!hnsw_index_) return 0;

    size_t vector_storage = hnsw_index_->ntotal * dimension_ * sizeof(float);
    size_t graph_storage = hnsw_index_->ntotal * M_ * 2 * sizeof(faiss::idx_t);

    return vector_storage + graph_storage;
}

void IndependentHNSWIndex::save_components(const std::string& base_dir) const {
    if (!hnsw_index_) {
        return;
    }

    std::filesystem::path dir(base_dir);
    std::filesystem::create_directories(dir);

    std::filesystem::path vec_path = dir / "vectors.bin";
    std::ofstream vec_out(vec_path, std::ios::binary);
    if (!vec_out) {
        std::cerr << "Failed to open " << vec_path << " for writing vectors" << std::endl;
    } else {
        int32_t dim = dimension_;
        faiss::idx_t count = hnsw_index_->ntotal;
        vec_out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        vec_out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        std::vector<float> buffer;
        if (count > 0) {
            buffer.resize(static_cast<size_t>(count) * static_cast<size_t>(dimension_));
            hnsw_index_->reconstruct_n(0, count, buffer.data());
            vec_out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
        }
        vec_out.close();

        std::ofstream meta(vec_path.string() + ".meta", std::ios::binary);
        if (!meta) {
            std::cerr << "Failed to open " << vec_path << ".meta for writing metadata" << std::endl;
        } else {
            meta.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            meta.write(reinterpret_cast<const char*>(&count), sizeof(count));
            for (const auto& [doc, block] : doc_block_ids_) {
                meta.write(reinterpret_cast<const char*>(&doc), sizeof(doc));
                meta.write(reinterpret_cast<const char*>(&block), sizeof(block));
            }
        }
    }

    std::filesystem::path graph_path = dir / "graph.bin";
    std::ofstream graph_out(graph_path, std::ios::binary);
    if (!graph_out) {
        std::cerr << "Failed to open " << graph_path << " for writing graph" << std::endl;
        return;
    }

    const faiss::HNSW& hnsw = hnsw_index_->hnsw;
    uint32_t magic = 0x574E5348;
    uint32_t version = 1;
    int64_t ntotal = hnsw_index_->ntotal;
    int32_t entry_point = hnsw.entry_point;
    int32_t max_level = hnsw.max_level;

    graph_out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    graph_out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    graph_out.write(reinterpret_cast<const char*>(&ntotal), sizeof(ntotal));
    graph_out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
    graph_out.write(reinterpret_cast<const char*>(&max_level), sizeof(max_level));

    size_t offsets_size = hnsw.offsets.size();
    graph_out.write(reinterpret_cast<const char*>(&offsets_size), sizeof(offsets_size));
    if (offsets_size > 0) {
        graph_out.write(reinterpret_cast<const char*>(hnsw.offsets.data()), offsets_size * sizeof(size_t));
    }

    size_t neighbors_size = hnsw.neighbors.size();
    graph_out.write(reinterpret_cast<const char*>(&neighbors_size), sizeof(neighbors_size));
    if (neighbors_size > 0) {
        graph_out.write(reinterpret_cast<const char*>(hnsw.neighbors.data()),
                        neighbors_size * sizeof(faiss::HNSW::storage_idx_t));
    }

    size_t levels_size = hnsw.levels.size();
    graph_out.write(reinterpret_cast<const char*>(&levels_size), sizeof(levels_size));
    if (levels_size > 0) {
        graph_out.write(reinterpret_cast<const char*>(hnsw.levels.data()), levels_size * sizeof(int));
    }

    size_t assign_size = hnsw.assign_probas.size();
    graph_out.write(reinterpret_cast<const char*>(&assign_size), sizeof(assign_size));
    if (assign_size > 0) {
        graph_out.write(reinterpret_cast<const char*>(hnsw.assign_probas.data()), assign_size * sizeof(double));
    }

    size_t cum_size = hnsw.cum_nneighbor_per_level.size();
    graph_out.write(reinterpret_cast<const char*>(&cum_size), sizeof(cum_size));
    if (cum_size > 0) {
        graph_out.write(reinterpret_cast<const char*>(hnsw.cum_nneighbor_per_level.data()),
                        cum_size * sizeof(int));
    }

    graph_out.close();
}

bool IndependentHNSWIndex::load_components(const std::string& base_dir) {
    std::filesystem::path dir(base_dir);
    std::filesystem::path vec_path = dir / "vectors.bin";
    std::ifstream vec_in(vec_path, std::ios::binary);
    if (!vec_in) {
        return false;
    }

    int32_t dim = 0;
    faiss::idx_t count = 0;
    vec_in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    vec_in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!vec_in) {
        return false;
    }

    std::vector<float> buffer(static_cast<size_t>(count) * static_cast<size_t>(dim));
    if (!buffer.empty()) {
        vec_in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
        if (!vec_in) {
            return false;
        }
    }
    vec_in.close();

    std::ifstream meta(vec_path.string() + ".meta", std::ios::binary);
    if (!meta) {
        return false;
    }
    int32_t meta_dim = 0;
    faiss::idx_t meta_count = 0;
    meta.read(reinterpret_cast<char*>(&meta_dim), sizeof(meta_dim));
    meta.read(reinterpret_cast<char*>(&meta_count), sizeof(meta_count));
    if (!meta || meta_dim != dim || meta_count != count) {
        return false;
    }
    std::vector<DocBlockId> doc_map(static_cast<size_t>(meta_count));
    for (size_t i = 0; i < doc_map.size(); ++i) {
        meta.read(reinterpret_cast<char*>(&doc_map[i].first), sizeof(doc_map[i].first));
        meta.read(reinterpret_cast<char*>(&doc_map[i].second), sizeof(doc_map[i].second));
        if (!meta) {
            return false;
        }
    }

    hnsw_index_ = std::make_unique<faiss::IndexHNSWFlat>(dim, M_, metric_);
    hnsw_index_->hnsw.efConstruction = efConstruction_;
    doc_block_ids_ = std::move(doc_map);

    if (normalize_ && count > 0) {
        normalize_buffer(buffer.data(), count, dim);
    }

    if (count > 0) {
        hnsw_index_->add(count, buffer.data());
    }

    std::filesystem::path graph_path = dir / "graph.bin";
    std::ifstream graph_in(graph_path, std::ios::binary);
    if (!graph_in) {
        return false;
    }

    uint32_t magic = 0;
    graph_in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    bool has_header = (magic == 0x574E5348);
    uint32_t version = 0;
    int64_t ntotal = 0;
    int32_t entry_point = 0;
    int32_t max_level = 0;

    if (has_header) {
        graph_in.read(reinterpret_cast<char*>(&version), sizeof(version));
        graph_in.read(reinterpret_cast<char*>(&ntotal), sizeof(ntotal));
        graph_in.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
        graph_in.read(reinterpret_cast<char*>(&max_level), sizeof(max_level));
    } else {
        uint32_t high = 0;
        graph_in.read(reinterpret_cast<char*>(&high), sizeof(high));
        ntotal = (static_cast<int64_t>(high) << 32) | magic;
    }

    size_t offsets_size = 0;
    graph_in.read(reinterpret_cast<char*>(&offsets_size), sizeof(offsets_size));
    std::vector<size_t> offsets(offsets_size);
    if (offsets_size > 0) {
        graph_in.read(reinterpret_cast<char*>(offsets.data()), offsets_size * sizeof(size_t));
    }

    size_t neighbors_size = 0;
    graph_in.read(reinterpret_cast<char*>(&neighbors_size), sizeof(neighbors_size));
    std::vector<faiss::HNSW::storage_idx_t> neighbors(neighbors_size);
    if (neighbors_size > 0) {
        graph_in.read(reinterpret_cast<char*>(neighbors.data()), neighbors_size * sizeof(faiss::HNSW::storage_idx_t));
    }

    size_t levels_size = 0;
    graph_in.read(reinterpret_cast<char*>(&levels_size), sizeof(levels_size));
    std::vector<int> levels(levels_size);
    if (levels_size > 0) {
        graph_in.read(reinterpret_cast<char*>(levels.data()), levels_size * sizeof(int));
    }

    size_t assign_size = 0;
    graph_in.read(reinterpret_cast<char*>(&assign_size), sizeof(assign_size));
    std::vector<double> assign(assign_size);
    if (assign_size > 0) {
        graph_in.read(reinterpret_cast<char*>(assign.data()), assign_size * sizeof(double));
    }

    size_t cum_size = 0;
    graph_in.read(reinterpret_cast<char*>(&cum_size), sizeof(cum_size));
    std::vector<int> cum(cum_size);
    if (cum_size > 0) {
        graph_in.read(reinterpret_cast<char*>(cum.data()), cum_size * sizeof(int));
    }

    if (!graph_in) {
        return false;
    }

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

    hnsw_index_->ntotal = ntotal;
    apply_colocated_layout();
    return true;
}

} // namespace pointer_benchmark
