#ifndef SHARED_VECTOR_TABLE_H
#define SHARED_VECTOR_TABLE_H

#include <vector>
#include <map>
#include <memory>
#include <pqxx/pqxx>
#include <faiss/MetricType.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>

namespace pointer_benchmark {

/**
 * Unified vector table storing all vectors from all partitions.
 * Pointer-based HNSW indices access vectors through this table to avoid duplication.
 */
class SharedVectorTable {
public:
    using DocBlockId = std::pair<int, int>;  // (document_id, block_id)

    struct DocBlockIdHash {
        size_t operator()(const DocBlockId& id) const noexcept {
            return (static_cast<size_t>(id.first) << 32) ^ static_cast<size_t>(id.second);
        }
    };

    SharedVectorTable(int dimension,
                      faiss::MetricType metric = faiss::METRIC_L2,
                      bool normalize = false);

    /**
     * Load all vectors from database into the shared table.
     * @param conn_info PostgreSQL connection string
     * @return Number of vectors loaded
     */
    size_t load_from_database(const std::string& conn_info);

    /**
     * Get vector by (document_id, block_id).
     * @return Pointer to vector data (dimension floats), nullptr if not found
     */
    const float* get_vector(int doc_id, int block_id) const;

    /**
     * Get internal index for (document_id, block_id).
     * Used by HNSW indices to map global IDs to local indices.
     * @return Internal index, or -1 if not found
     */
    faiss::idx_t get_internal_index(int doc_id, int block_id) const;

    /**
     * Get (document_id, block_id) from internal index.
     */
    DocBlockId get_doc_block_id(faiss::idx_t internal_idx) const;

    /**
     * Get the underlying Faiss IndexFlat (for sharing with HNSW).
     * @return Pointer to the IndexFlat containing all vectors
     */
    faiss::IndexFlat* get_index_flat() { return index_flat_.get(); }
    const faiss::IndexFlat* get_index_flat() const { return index_flat_.get(); }

    size_t size() const { return id_to_index_.size(); }
    int dimension() const { return dimension_; }

    faiss::MetricType metric() const { return metric_; }
    
    bool normalized() const { return normalize_; }
    void flush_cache() const;

    std::unique_ptr<SharedVectorTable> clone() const;

    void save_vectors(const std::string& path) const;
    bool load_vectors(const std::string& path);

private:
    int dimension_;
    faiss::MetricType metric_;
    bool normalize_;

    // The actual vector storage (Faiss IndexFlat)
    std::unique_ptr<faiss::IndexFlat> index_flat_;

    // Mapping: (document_id, block_id) -> internal index in index_flat_
    std::map<DocBlockId, faiss::idx_t> id_to_index_;

    // Reverse mapping: internal index -> (document_id, block_id)
    std::vector<DocBlockId> index_to_id_;
};

} // namespace pointer_benchmark

#endif // SHARED_VECTOR_TABLE_H
