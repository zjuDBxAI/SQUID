#include "global_hnsw_index.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace pointer_benchmark {


GlobalHNSWIndex::GlobalHNSWIndex(IndexMode mode,
                                 std::shared_ptr<SharedVectorTable> shared_table,
                                 int M,
                                 int efConstruction)
    : mode_(mode),
      M_(M),
      efConstruction_(efConstruction),
      ef_search_(100),
      shared_table_(std::move(shared_table))
{
    if (!shared_table_) {
        throw std::invalid_argument("A non-null shared_table is required for both modes.");
    }

    if (mode_ == IndexMode::SHARED_POINTER) {
        auto hnsw = std::make_unique<faiss::IndexHNSW>(shared_table_->get_index_flat(), M_);
        hnsw->hnsw.efConstruction = efConstruction_;
        hnsw->own_fields = false;
        hnsw_index_.emplace<std::unique_ptr<faiss::IndexHNSW>>(std::move(hnsw));
    } else { // INDEPENDENT mode
        const int dimension = shared_table_->get_index_flat()->d;
        auto hnsw = std::make_unique<faiss::IndexHNSWLib>(dimension, M_, faiss::METRIC_L2, efConstruction_, 100);
        hnsw_index_.emplace<std::unique_ptr<faiss::IndexHNSWLib>>(std::move(hnsw));
    }
}

size_t GlobalHNSWIndex::ntotal() const {
    return std::visit([](auto&& index) -> size_t {
        if (index) {
            return index->ntotal;
        }
        return 0;
    }, hnsw_index_);
}

size_t GlobalHNSWIndex::memory_bytes() const {
    return std::visit([this](auto&& index) -> size_t {
        using T = std::decay_t<decltype(index)>;
        if (!index) {
            return 0;
        }

        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
            // SHARED_POINTER mode: only count graph, vectors are in shared_table_
            // Graph memory = HNSW structure
            const faiss::HNSW& hnsw = index->hnsw;
            size_t graph_bytes = 0;

            // neighbors array
            graph_bytes += hnsw.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t);
            // offsets array
            graph_bytes += hnsw.offsets.size() * sizeof(size_t);
            // levels array
            graph_bytes += hnsw.levels.size() * sizeof(int);
            // assign_probas array
            graph_bytes += hnsw.assign_probas.size() * sizeof(double);
            // cum_nneighbor_per_level array
            graph_bytes += hnsw.cum_nneighbor_per_level.size() * sizeof(int);
            // storage_ids array
            graph_bytes += index->storage_ids.size() * sizeof(faiss::idx_t);

            return graph_bytes;
        } else {
            // INDEPENDENT mode: count everything (vectors + graph)
            return index->allocated_bytes();
        }
    }, hnsw_index_);
}

void GlobalHNSWIndex::build() {
    std::visit([this](auto&& index) {
        using T = std::decay_t<decltype(index)>;
        faiss::IndexFlat* storage = shared_table_->get_index_flat();
        faiss::idx_t ntotal = storage->ntotal;

        if (ntotal == 0) return;

        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
            std::vector<faiss::idx_t> all_indices(ntotal);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            index->add_from_storage_ids(ntotal, all_indices.data());
        } else {
            const float* all_vectors = storage->get_xb();
            index->add(ntotal, all_vectors);
            // Build doc_block_ids_ from shared_table_
            doc_block_ids_.clear();
            doc_block_ids_.reserve(ntotal);
            for (faiss::idx_t i = 0; i < ntotal; ++i) {
                doc_block_ids_.push_back(shared_table_->get_doc_block_id(i));
            }
        }
    }, hnsw_index_);
}

void GlobalHNSWIndex::set_ef_search(int ef) {
    ef_search_ = ef;
    std::visit([ef](auto&& index) {
        using T = std::decay_t<decltype(index)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
            index->hnsw.efSearch = ef;
        } else if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSWLib>>) {
            index->set_efSearch(ef);
        }
    }, hnsw_index_);
}

const faiss::IndexHNSW* GlobalHNSWIndex::get_index() const {
    if (auto* p_hnsw_ptr = std::get_if<std::unique_ptr<faiss::IndexHNSW>>(&hnsw_index_)) {
        return p_hnsw_ptr->get();
    }
    return nullptr; // INDEPENDENT
}

faiss::IndexHNSW* GlobalHNSWIndex::get_index() {
    if (auto* p_hnsw_ptr = std::get_if<std::unique_ptr<faiss::IndexHNSW>>(&hnsw_index_)) {
        return p_hnsw_ptr->get();
    }
    return nullptr; // INDEPENDENT 
}


std::vector<GlobalHNSWIndex::DocBlockId> GlobalHNSWIndex::search_filtered(
        const float* query,
        int k,
        const std::vector<faiss::idx_t>& allowed_ids, 
        const std::vector<uint8_t>& allowed_mask,
        std::vector<float>& temp_dist,
        std::vector<faiss::idx_t>& temp_idx) const
{
    std::vector<DocBlockId> results;
    if (k <= 0 || allowed_ids.empty()) {
        return results;
    }

    std::visit([&](auto&& index) {
        const int ntotal_val = static_cast<int>(index->ntotal);
        const int search_k = std::min(ntotal_val, std::max(k, ef_search_));
        
        if (static_cast<int>(temp_dist.size()) < search_k) {
            temp_dist.resize(search_k);
            temp_idx.resize(search_k);
        }

        using T = std::decay_t<decltype(index)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
            index->hnsw.efSearch = std::max(ef_search_, search_k);
        } else {
            index->set_efSearch(std::max(ef_search_, search_k));
        }
        index->search(1, query, search_k, temp_dist.data(), temp_idx.data());

        std::unordered_set<DocBlockId, SharedVectorTable::DocBlockIdHash> seen;
        results.reserve(k);
        for (int i = 0; i < search_k && results.size() < static_cast<size_t>(k); ++i) {
            faiss::idx_t id = temp_idx[i];
            if (id < 0) continue;

            if (id < static_cast<faiss::idx_t>(allowed_mask.size()) && allowed_mask[id]) {
                DocBlockId doc_block;
                if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
                    doc_block = shared_table_->get_doc_block_id(id);
                } else {
                    doc_block = doc_block_ids_[id];
                }
                if (seen.insert(doc_block).second) {
                    results.push_back(doc_block);
                }
            }
        }
    }, hnsw_index_);

    return results;
}

void GlobalHNSWIndex::save_graph(const std::string& path) const {
    std::visit([&](auto&& index) {
        using T = std::decay_t<decltype(index)>;

        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
            std::ofstream out(path, std::ios::binary);
            if (!out) {
                std::cerr << "Failed to open " << path << " for writing global HNSW graph" << std::endl;
                return;
            }

            const faiss::HNSW& hnsw = index->hnsw;
            uint32_t magic = 0x574E5348;
            uint32_t version = 1;
            int64_t ntotal = index->ntotal;
            int32_t entry_point = hnsw.entry_point;
            int32_t max_level = hnsw.max_level;

            out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            out.write(reinterpret_cast<const char*>(&version), sizeof(version));
            out.write(reinterpret_cast<const char*>(&ntotal), sizeof(ntotal));
            out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
            out.write(reinterpret_cast<const char*>(&max_level), sizeof(max_level));

            size_t storage_ids_size = index->storage_ids.size();
            out.write(reinterpret_cast<const char*>(&storage_ids_size), sizeof(storage_ids_size));
            if (storage_ids_size > 0) {
                out.write(reinterpret_cast<const char*>(index->storage_ids.data()),
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

            out.close();
        } else if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSWLib>>) {
            if (!index->save(path.c_str())) {
                std::cerr << "Failed to save HNSWLib index to " << path << std::endl;
                return;
            }

            std::string meta_path = path + ".meta";
            std::ofstream meta_out(meta_path, std::ios::binary);
            if (!meta_out) {
                std::cerr << "Failed to open " << meta_path << " for writing metadata" << std::endl;
                return;
            }

            uint64_t count = doc_block_ids_.size();
            meta_out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            if (count > 0) {
                meta_out.write(reinterpret_cast<const char*>(doc_block_ids_.data()),
                            count * sizeof(DocBlockId));
            }
        }
    }, hnsw_index_);
}

bool GlobalHNSWIndex::load_graph(const std::string& path) {
    return std::visit([&](auto&& index) -> bool {
        using T = std::decay_t<decltype(index)>;

        if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSW>>) {
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

            if (!in) {
                return false;
            }

            // Create new index and assign to variant
            auto new_index = std::make_unique<faiss::IndexHNSW>(shared_table_->get_index_flat(), M_);
            new_index->hnsw.efConstruction = efConstruction_;
            new_index->own_fields = false;
            new_index->storage_ids = std::move(storage_ids);
            new_index->use_storage_ids = !new_index->storage_ids.empty();
            new_index->ntotal = ntotal;

            faiss::HNSW& hnsw = new_index->hnsw;
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

            // Replace the variant with the new index
            const_cast<std::variant<std::unique_ptr<faiss::IndexHNSW>, std::unique_ptr<faiss::IndexHNSWLib>>&>(
                const_cast<GlobalHNSWIndex*>(this)->hnsw_index_
            ) = std::move(new_index);

            return true;
        } else if constexpr (std::is_same_v<T, std::unique_ptr<faiss::IndexHNSWLib>>) {
            if (!index->load(path.c_str())) {
                std::cerr << "Failed to load HNSWLib index from " << path << std::endl;
                return false;
            }

            std::string meta_path = path + ".meta";
            std::ifstream meta_in(meta_path, std::ios::binary);
            if (!meta_in) {
                std::cerr << "Failed to open metadata file " << meta_path << std::endl;
                return false;
            }

            uint64_t count = 0;
            meta_in.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (count > 0) {
                const_cast<GlobalHNSWIndex*>(this)->doc_block_ids_.resize(count);
                meta_in.read(reinterpret_cast<char*>(const_cast<GlobalHNSWIndex*>(this)->doc_block_ids_.data()),
                             count * sizeof(DocBlockId));
            }

            if (!meta_in) {
                std::cerr << "Error reading metadata from " << meta_path << std::endl;
                return false;
            }

            if (static_cast<uint64_t>(index->ntotal) != count) {
                std::cerr << "Warning: index ntotal (" << index->ntotal
                          << ") does not match metadata count (" << count << ")" << std::endl;
            }
            return true;
        }
        return false;
    }, hnsw_index_);
}

} // namespace pointer_benchmark
