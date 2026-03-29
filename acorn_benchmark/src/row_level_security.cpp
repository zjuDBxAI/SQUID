#include "acorn_search.h"
#include "benchmark_utils.h" // For compute_groundtruth and helper functions
#include <faiss/IndexACORN.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <thread>
#include "index_creation.h"

// Implementation of acorn_search
std::tuple<double, double> hnsw_search(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    const HNSWIndexWithMetadata &hnsw_index_metadata,
    std::vector<char> &fields_map
){
    double total_time = 0.0;
    double avg_recall = 0.0;

    const auto &hnsw_index = hnsw_index_metadata.index;
    const auto &document_block_map = hnsw_index_metadata.document_block_map;

    // Precompute Ground Truth: list of <document_id, block_id> for each query
    auto ground_truth = compute_groundtruth(queries, conn_info);

    size_t d = queries[0].query_vector.size(); // Dimensionality of query vectors
    size_t k = queries[0].topk; // Top-k results
    size_t nq = queries.size(); // Number of queries
    size_t N = document_block_map.size(); // Number of blocks (vectors)


    // Stores results for all queries
    std::vector<std::pair<int, int>> all_retrieved_results;


    for (size_t xq = 0; xq < nq; ++xq) {
        const auto& query = queries[xq];

        // Extract the specific slice of fields_map for this query
        std::vector<char> single_query_fields_map(fields_map.begin() + xq * N, fields_map.begin() + (xq + 1) * N);

        // Prepare output containers for this query
        std::vector<faiss::idx_t> indices(k); // Top-k indices for the query
        std::vector<float> distances(k); // Top-k distances for the query

        // Perform single-query ACORN search
        auto start = std::chrono::high_resolution_clock::now();
        hnsw_index->hnsw.efSearch = 300;
        hnsw_index->search(
            1, // Single query
            query.query_vector.data(), // Query vector
            k, // Top-k results
            distances.data(), // Output distances
            indices.data() // Output indices
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        total_time += duration.count();

        // Process results for this query
        for (size_t i = 0; i < k; ++i) {
            if (indices[i] >= 0 && static_cast<size_t>(indices[i]) < document_block_map.size()) {
                // Ensure index is valid
                all_retrieved_results.push_back(document_block_map[indices[i]]);
            } else {
                all_retrieved_results.emplace_back(-1, -1); // Placeholder for invalid result
            }
        }
    }

    // Compute recall for all queries
    avg_recall = compute_recall(ground_truth, all_retrieved_results, k);

    // Compute average time per query
    total_time /= queries.size();

    return {total_time, avg_recall};
}