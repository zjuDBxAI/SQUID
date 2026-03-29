#ifndef INDEX_CREATION_H
#define INDEX_CREATION_H

#include <string>
#include <unordered_map>
#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>

// Index container for both ACORN and HNSW
struct PartitionIndex {
    std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index;
    std::unique_ptr<faiss::IndexACORNFlat> acorn_index;
    std::vector<std::pair<int, int> > document_block_map; // <document_id, block_id>
};

struct ACORNIndexWithMetadata {
    std::unique_ptr<faiss::IndexACORNFlat> index;
    std::vector<std::pair<int, int> > document_block_map; // <document_id, block_id>
};

struct HNSWIndexWithMetadata {
    std::unique_ptr<faiss::IndexHNSWFlat> index;
    std::vector<std::pair<int, int> > document_block_map; // <document_id, block_id>
};

HNSWIndexWithMetadata create_hnsw_index(
    const std::string &conn_info
);

// Create ACORN index for documentblocks
ACORNIndexWithMetadata create_acorn_index(const std::string &conn_info);

// Create dynamic partition indices (ACORN or HNSW based on role-document logic)
std::unordered_map<std::string, PartitionIndex> create_dynamic_partition_indices(
    const std::string &conn_info
);

// Create ACORN index for documentblocks
void try_create_acorn_index(const std::string &conn_info);

// Create dynamic partition indices (ACORN or HNSW based on role-document logic)
void try_create_dynamic_partition_indices(
    const std::string &conn_info
);

#endif // INDEX_CREATION_H
