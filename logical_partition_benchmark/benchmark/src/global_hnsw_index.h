#pragma once

#include "shared_vector_table.h"
#include <faiss/IndexHNSW.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <faiss/IndexHNSWLib.h>
#include <faiss/impl/HNSW.h>
#include <faiss/MetricType.h>

namespace pointer_benchmark {

enum class IndexMode {
    SHARED_POINTER, 
    INDEPENDENT     
};

class GlobalHNSWIndex {
public:
    using DocBlockId = std::pair<int, int>;

    // GlobalHNSWIndex(std::shared_ptr<SharedVectorTable> shared_table,
    //                 int M,
    //                 int efConstruction);
    GlobalHNSWIndex(IndexMode mode,
                    std::shared_ptr<SharedVectorTable> shared_table,
                    int M,
                    int efConstruction);

    void build();

    void set_ef_search(int ef);

    std::vector<DocBlockId> search_filtered(
            const float* query,
            int k,
            const std::vector<faiss::idx_t>& allowed_ids,
            const std::vector<uint8_t>& allowed_mask,
            std::vector<float>& temp_dist,
            std::vector<faiss::idx_t>& temp_idx) const;


    const faiss::IndexHNSW* get_index() const;
    faiss::IndexHNSW* get_index();

    void save_graph(const std::string& path) const;
    bool load_graph(const std::string& path);

    size_t ntotal() const;
    size_t memory_bytes() const;
private:
    IndexMode mode_;
    int dimension_;
    int M_;
    int efConstruction_;
    int ef_search_;

    std::shared_ptr<SharedVectorTable> shared_table_;

    std::vector<DocBlockId> doc_block_ids_;

    std::variant<
        std::unique_ptr<faiss::IndexHNSW>,
        std::unique_ptr<faiss::IndexHNSWLib>
    > hnsw_index_;
    };
} // namespace pointer_benchmark
