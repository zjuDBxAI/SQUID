#ifndef POINTER_HNSW_INDEX_H
#define POINTER_HNSW_INDEX_H

#include "shared_vector_table.h"
#include <faiss/IndexHNSW.h>
#include <memory>
#include <vector>
#include <string>

namespace pointer_benchmark {

/**
 * Pointer-based HNSW index that shares vectors with a global table.
 * The HNSW graph only stores IDs, vectors are accessed via SharedVectorTable.
 */
class PointerHNSWIndex {
public:
    using DocBlockId = std::pair<int, int>;

    /**
     * Constructor.
     * @param shared_table Shared vector table (must outlive this index)
     * @param M HNSW M parameter
     * @param efConstruction HNSW efConstruction parameter
     */
    PointerHNSWIndex(
        std::shared_ptr<SharedVectorTable> shared_table,
        int M = 32,
        int efConstruction = 200
    );

    /**
     * Build HNSW index for a specific partition.
     * @param conn_info PostgreSQL connection string
     * @param partition_table_name Name of the partition table (e.g., "documentblocks_role_1")
     */
    void build_from_partition(
        const std::string& conn_info,
        const std::string& partition_table_name
    );

    /**
     * Search for k nearest neighbors.
     * @param query Query vector
     * @param k Number of neighbors
     * @return Vector of (document_id, block_id) pairs
     */
    std::vector<DocBlockId> search(const float* query, int k);

    /**
     * Set efSearch parameter.
     */
    void set_ef_search(int ef) { ef_search_ = ef; }

    /**
     * Force legacy behaviour (copy vectors into the HNSW storage).
     * Default is false, meaning we only build the graph over shared vectors.
     */
    void set_force_copy_mode(bool force_copy) { force_copy_vectors_ = force_copy; }

    /**
     * Get number of vectors in this partition index.
     */
    size_t size() const { return doc_block_ids_.size(); }

    faiss::IndexHNSW* get_hnsw_index() { return hnsw_index_.get(); }
    const std::vector<DocBlockId>& doc_blocks() const { return doc_block_ids_; }

    void save_graph(const std::string& path) const;
    bool load_graph(const std::string& path);

private:
    std::shared_ptr<SharedVectorTable> shared_table_;
    std::unique_ptr<faiss::IndexHNSW> hnsw_index_;  // Use IndexHNSW with shared storage

    // Mapping: local HNSW index -> (document_id, block_id)
    std::vector<DocBlockId> doc_block_ids_;

    int M_;
    int efConstruction_;
    int ef_search_;

    bool force_copy_vectors_;
};

} // namespace pointer_benchmark

#endif // POINTER_HNSW_INDEX_H
