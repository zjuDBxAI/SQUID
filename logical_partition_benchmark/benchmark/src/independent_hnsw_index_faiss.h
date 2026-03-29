#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>

namespace pointer_benchmark {

// Independent HNSW index: each partition owns its own complete HNSW index with vector storage
// This is the baseline for comparison - vectors are duplicated across partitions
class IndependentHNSWIndex {
public:
    using DocBlockId = std::pair<int, int>;  // (document_id, block_id)

    struct ColocatedSlice {
        const uint8_t* vector_data = nullptr;
        size_t vector_bytes = 0;
        const faiss::HNSW::storage_idx_t* neighbor_data = nullptr;
        size_t neighbor_count = 0;
    };

    IndependentHNSWIndex(int dimension,
                         int M,
                         int efConstruction,
                         faiss::MetricType metric = faiss::METRIC_L2,
                         bool normalize = false,
                         bool colocate_storage = true);

    // Build index from a partition table
    void build_from_partition(
        const std::string& conn_info,
        const std::string& partition_table_name
    );

    // Search for k nearest neighbors
    std::vector<DocBlockId> search(const float* query, int k);

    // Get/set efSearch parameter
    void set_ef_search(int ef) { ef_search_ = ef; }
    int get_ef_search() const { return ef_search_; }

    // Get memory usage
    size_t memory_bytes() const;

    // Get number of vectors in this index
    size_t size() const { return doc_block_ids_.size(); }

    faiss::IndexHNSW* get_hnsw_index() { return hnsw_index_.get(); }
    const faiss::IndexHNSW* get_hnsw_index() const { return hnsw_index_.get(); }

    void save_components(const std::string& base_dir) const;
    bool load_components(const std::string& base_dir);

    void set_colocated_storage(bool enable) { colocate_storage_ = enable; }
    bool colocated_storage() const { return colocate_storage_; }

    ColocatedSlice get_colocated_slice() const;
    void rebind_colocated_slice(const std::shared_ptr<std::vector<uint8_t>>& buffer,
                                size_t vector_offset_bytes,
                                size_t neighbor_offset_bytes,
                                size_t neighbor_count);

private:
    void apply_colocated_layout();

    int dimension_;
    int M_;
    int efConstruction_;
    int ef_search_;
    faiss::MetricType metric_;
    bool normalize_;
    bool colocate_storage_;

    // Each partition owns its complete HNSW index (including vector storage)
    std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index_;

    // Map local index to (document_id, block_id)
    std::vector<DocBlockId> doc_block_ids_;

    std::shared_ptr<std::vector<uint8_t>> colocated_buffer_;
    std::shared_ptr<faiss::MaybeOwnedVectorOwner> colocated_owner_;
};

} // namespace pointer_benchmark
