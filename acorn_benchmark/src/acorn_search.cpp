#include "acorn_search.h"
#include "benchmark_utils.h" // For compute_groundtruth and helper functions
#include <faiss/IndexACORN.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <thread>
#include <filesystem>
#include <faiss/index_io.h>
#include <pqxx/pqxx>

#include "index_creation.h"


// Implementation of acorn_search
std::tuple<double, double> acorn_search(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    const ACORNIndexWithMetadata &acorn_index_metadata,
    std::vector<char> &fields_map // fields_map provided as parameter
) {
    double total_time = 0.0;
    double avg_recall = 0.0;

    const auto &acorn_index = acorn_index_metadata.index;
    const auto &document_block_map = acorn_index_metadata.document_block_map;

    // Precompute Ground Truth: list of <document_id, block_id> for each query
    auto ground_truth = compute_groundtruth(queries, conn_info);

    size_t d = queries[0].query_vector.size(); // Dimensionality of query vectors
    size_t k = queries[0].topk; // Top-k results
    size_t nq = queries.size(); // Number of queries
    size_t N = document_block_map.size(); // Number of blocks (vectors)


    // Stores results for all queries
    std::vector<std::pair<int, int> > all_retrieved_results;

    std::vector<double> time;
    for (size_t xq = 0; xq < nq; ++xq) {
        const auto &query = queries[xq];

        // Extract the specific slice of fields_map for this query
        std::vector<char> single_query_fields_map(fields_map.begin() + xq * N, fields_map.begin() + (xq + 1) * N);

        // Prepare output containers for this query
        std::vector<faiss::idx_t> indices(k); // Top-k indices for the query
        std::vector<float> distances(k); // Top-k distances for the query
        auto query_vector_data = query.query_vector.data();
        auto single_query_fields_map_data = single_query_fields_map.data();

        //warm up
        acorn_index->search(
            1, // Single query
            query_vector_data, // Query vector
            k, // Top-k results
            distances.data(), // Output distances
            indices.data(), // Output indices
            single_query_fields_map_data // Filter for this query
        );
        // Perform single-query ACORN search
        auto start = std::chrono::high_resolution_clock::now();
        acorn_index->acorn.efSearch = 45;
        acorn_index->search(
            1, // Single query
            query_vector_data, // Query vector
            k, // Top-k results
            distances.data(), // Output distances
            indices.data(), // Output indices
            single_query_fields_map_data // Filter for this query
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        total_time += duration.count();
        time.push_back(duration.count());
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
    std::cout << "hhhhhhhhhhhhhh:    " << time[0] << std::endl;
    return {total_time, avg_recall};
}


std::tuple<double, double> acorn_search_with_reading_index(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    std::vector<char> &fields_map, // fields_map provided as a parameter
    int ef_search
) {
    double total_time = 0.0;
    double avg_recall = 0.0;

    // Precompute Ground Truth: list of <document_id, block_id> for each query
    auto ground_truth = compute_groundtruth(queries, conn_info);

    // Get the project root and index file path
    std::filesystem::path index_filename = get_index_storage_root() / "acorn_index.faiss";


    // Load document_block_map outside the loop since it's constant
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);
    std::vector<std::pair<int, int>> document_block_map;
    pqxx::result doc_res = txn.exec(
        "SELECT document_id, block_id FROM documentblocks ORDER BY document_id, block_id"
    );
    for (const auto &row : doc_res) {
        document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>());
    }
    txn.commit();

    // Validate document_block_map
    if (document_block_map.empty()) {
        throw std::runtime_error("No document-block mappings found in the database.");
    }

    // Store results for all queries
    std::vector<std::pair<int, int>> all_retrieved_results;
    std::vector<double> time; // Per-query time

    size_t d = queries[0].query_vector.size(); // Dimensionality of query vectors
    size_t k = queries[0].topk;                // Top-k results
    size_t nq = queries.size();                // Number of queries

    for (size_t xq = 0; xq < nq; ++xq) {
        const auto &query = queries[xq];

        // Dynamically load the index
        std::unique_ptr<faiss::IndexACORNFlat> acorn_index(
            dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index(index_filename.c_str()))
        );
        if (!acorn_index) {
            throw std::runtime_error("Failed to load ACORN index from file: " + index_filename.string());
        }

        // Extract the specific slice of fields_map for this query
        size_t N = acorn_index->ntotal; // Total number of blocks in the index
        std::vector<char> single_query_fields_map(fields_map.begin() + xq * N, fields_map.begin() + (xq + 1) * N);

        // Prepare result containers for this query
        std::vector<faiss::idx_t> indices(k);  // Top-k indices
        std::vector<float> distances(k);      // Top-k distances
        auto query_vector_data = query.query_vector.data();
        auto single_query_fields_map_data = single_query_fields_map.data();

        // Warmup phase
        acorn_index->search(
            1,                                // Single query
            query_vector_data,                // Query vector
            k,                                // Top-k results
            distances.data(),                 // Output distances
            indices.data(),                   // Output indices
            single_query_fields_map_data      // Filter for this query
        );

        // Measure query time
        auto start = std::chrono::high_resolution_clock::now();
        acorn_index->acorn.efSearch = ef_search;     // Adjust the search parameter as needed
        acorn_index->search(
            1,
            query_vector_data,
            k,
            distances.data(),
            indices.data(),
            single_query_fields_map_data
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        total_time += duration.count();
        time.push_back(duration.count());

        // Process query results
        for (size_t i = 0; i < k; ++i) {
            if (indices[i] >= 0 && static_cast<size_t>(indices[i]) < document_block_map.size()) {
                // Map index to document-block mapping
                all_retrieved_results.emplace_back(document_block_map[indices[i]]);
            } else {
                all_retrieved_results.emplace_back(-1, -1); // Placeholder for invalid result
            }
        }

        // Release the index memory after each query
        acorn_index.reset();
    }

    // Compute recall
    avg_recall = compute_recall(ground_truth, all_retrieved_results, k);

    // Compute average time per query
    total_time /= queries.size();
    std::cout << "First query time: " << time[0] << std::endl;

    return {total_time, avg_recall};
}
