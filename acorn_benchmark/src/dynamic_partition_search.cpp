#include "dynamic_partition_search.h"
#include "benchmark_utils.h"
#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <faiss/index_io.h>
#include <pqxx/pqxx>
#include <faiss/impl/platform_macros.h>

// Implementation of benchmark_dynamic_partition_search
std::tuple<double, double> benchmark_dynamic_partition_search(
    const std::vector<Query> &queries,
    const std::unordered_map<std::string, PartitionIndex> &partition_indices,
    const std::string &conn_info
) {
    double total_time = 0.0;
    double avg_recall = 0.0;
    int topk = queries[0].topk;

    // Precompute Ground Truth for all queries
    auto ground_truth = compute_groundtruth(queries, conn_info);

    // Stores results for all queries
    std::vector<std::pair<int, int> > all_retrieved_results;
    std::vector<double> time;
    // Iterate over each query
    for (const auto &query: queries) {
        // Step 1: Retrieve relevant partitions for the current query
        pqxx::connection conn(conn_info);
        pqxx::work txn(conn);

        std::string query_str =
                "SELECT partition_id FROM RolePartitions rp "
                "JOIN UserRoles ur ON rp.role_id = ur.role_id "
                "WHERE ur.user_id = " + txn.quote(query.user_id) + ";";

        pqxx::result partition_res = txn.exec(query_str);
        txn.commit();

        std::set<std::string> relevant_partitions;
        for (const auto &row: partition_res) {
            relevant_partitions.insert(std::string("documentblocks_partition_") + row[0].c_str());
        }

        // Step 2: Search through relevant partitions
        std::vector<std::pair<float, std::pair<int, int> > > combined_results; // Distance and <document_id, block_id>
        for (const auto &partition_name: relevant_partitions) {
            const auto &partition_index = partition_indices.at(partition_name);

            // Prepare output containers for this partition
            std::vector<float> distances(query.topk);
            std::vector<faiss::idx_t> indices(query.topk);
            std::vector<std::pair<float, std::pair<int, int> > > partition_results;

            double partition_search_time = 0.0; // Time for this partition's index search

            if (partition_index.acorn_index) {
                // Generate fields_map for this partition
                auto fields_map = generate_fields_map(
                    conn_info, // Connection info for SQL queries
                    partition_name, // Current partition table name
                    query.user_id // User ID for the query
                );
                // warm up
                partition_index.acorn_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data(),
                    fields_map.data()
                );
                // Perform ACORN search (timed)
                auto start = std::chrono::high_resolution_clock::now();
                // partition_index.acorn_index->acorn.efSearch = 24;
                partition_index.acorn_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data(),
                    fields_map.data()
                );
                auto end = std::chrono::high_resolution_clock::now();
                partition_search_time = std::chrono::duration<double>(end - start).count();
            } else if (partition_index.hnsw_index) {
                //warm up
                partition_index.hnsw_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data()
                );
                // Perform HNSW search (timed)
                auto start = std::chrono::high_resolution_clock::now();
                // partition_index.hnsw_index->hnsw.efSearch = 200;
                partition_index.hnsw_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data()
                );
                auto end = std::chrono::high_resolution_clock::now();
                partition_search_time = std::chrono::duration<double>(end - start).count();
            }

            // Add this partition's search time to total_time
            total_time += partition_search_time;
            time.push_back(partition_search_time);
            // Convert results for this partition
            for (size_t i = 0; i < query.topk; ++i) {
                if (indices[i] >= 0 && static_cast<size_t>(indices[i]) < partition_index.document_block_map.size()) {
                    // Valid result
                    partition_results.emplace_back(
                        distances[i], // Distance
                        partition_index.document_block_map[indices[i]] // <document_id, block_id>
                    );
                } else {
                    // Invalid result, use placeholder
                    partition_results.emplace_back(
                        -1, std::make_pair(-1, -1));
                }
            }

            // Merge this partition's results into combined results
            combined_results.insert(
                combined_results.end(),
                partition_results.begin(),
                partition_results.end()
            );
        }

        // Step 3: Sort combined results and keep top-k
        std::sort(combined_results.begin(), combined_results.end(), [](const auto &a, const auto &b) {
            if (a.first == -1 && b.first == -1) return false;
            if (a.first == -1) return false; // Invalid results go last
            if (b.first == -1) return true;
            return a.first < b.first; // Sort by distance
        });

        combined_results.resize(query.topk);

        // Convert query results into global results
        for (const auto &result: combined_results) {
            all_retrieved_results.push_back(result.second);
        }
    }

    // Step 4: Compute recall
    avg_recall = compute_recall(ground_truth, all_retrieved_results, topk);

    // Compute average time per query
    total_time /= queries.size();

    return {total_time, avg_recall};
}

struct ResultComparator {
    bool operator()(const std::pair<float, std::pair<int, int>> &a,
                    const std::pair<float, std::pair<int, int>> &b) const {
        return a.second < b.second; // Only compare by `second`
    }
};

// Implementation of benchmark_dynamic_partition_search
std::tuple<double, double> benchmark_dynamic_partition_search_with_reading_index(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    int ef_search
) {
    double total_time = 0.0;
    double avg_recall = 0.0;
    int topk = queries[0].topk;

    // Precompute Ground Truth for all queries
    auto ground_truth = compute_groundtruth(queries, conn_info);

    // Cache all document_block_maps
    auto document_block_maps = fetch_all_document_block_maps(conn_info);

    // Stores results for all queries
    std::vector<std::pair<int, int> > all_retrieved_results;

    // Iterate over each query
    for (size_t query_idx = 0; query_idx < queries.size(); ++query_idx) {
        const auto &query = queries[query_idx];
        // Step 1: Retrieve relevant partitions for the current query
        pqxx::connection conn(conn_info);
        pqxx::work txn(conn);

        // Retrieve roles for the user
        std::string role_query =
            "SELECT role_id FROM UserRoles WHERE user_id = " + txn.quote(query.user_id) + ";";
        pqxx::result role_res = txn.exec(role_query);

        // Store roles in a sorted vector
        std::vector<int> user_roles;
        for (const auto &row : role_res) {
            user_roles.push_back(row[0].as<int>());
        }
        std::sort(user_roles.begin(), user_roles.end());

        // Retrieve partitions using the sorted roles
        std::string partition_query =
            "SELECT partition_id FROM CombRolePartitions WHERE comb_role = " +
            txn.quote(pqxx::to_string(user_roles)) + "::integer[];";
        pqxx::result partition_res = txn.exec(partition_query);

        txn.commit();

        std::set<std::string> relevant_partitions;
        for (const auto &row: partition_res) {
            relevant_partitions.insert(std::string("documentblocks_partition_") + row[0].c_str());
        }

        // Stores unique combined results sorted by distance
        std::set<std::pair<float, std::pair<int, int>>, ResultComparator> combined_results;
        for (const auto &partition_name: relevant_partitions) {
            // Define the index file path
            std::filesystem::path index_path =
                get_index_storage_root() / "dynamic_partition" / (partition_name + ".faiss");

            // Dynamically load index
            std::unique_ptr<faiss::Index> index(faiss::read_index(index_path.c_str()));
            if (!index) {
                throw std::runtime_error("Failed to load index for partition: " + partition_name);
            }

            // Check index type
            std::unique_ptr<faiss::IndexACORNFlat> acorn_index(dynamic_cast<faiss::IndexACORNFlat *>(index.get()));
            std::unique_ptr<faiss::IndexHNSWFlat> hnsw_index(dynamic_cast<faiss::IndexHNSWFlat *>(index.get()));

            index.release(); // Release ownership to specific type pointer

            // Retrieve pre-cached document_block_map
            const auto &document_block_map = document_block_maps.at(partition_name);

            const size_t expected_size = document_block_map.size();
            const size_t actual_size =
                acorn_index ? static_cast<size_t>(acorn_index->ntotal) : static_cast<size_t>(hnsw_index->ntotal);
            if (actual_size != expected_size) {
                std::cerr << "[WARN] Index/doc_map size mismatch for " << partition_name
                          << ": index has " << actual_size << " entries but table has "
                          << expected_size << ". Rebuild the index." << std::endl;
                if (actual_size < expected_size) {
                    // avoid out-of-bounds
                    std::cerr << "[WARN] Skipping partition due to mismatch." << std::endl;
                    continue;
                }
            }

            // Prepare output containers for this partition
            std::vector<float> distances(query.topk);
            std::vector<faiss::idx_t> indices(query.topk);
            std::vector<std::pair<float, std::pair<int, int> > > partition_results;

            double partition_search_time = 0.0; // Time for this partition's index search

            if (acorn_index) {
                // Generate fields_map for this partition
                auto fields_map = generate_fields_map(
                    conn_info, // Connection info for SQL queries
                    partition_name, // Current partition table name
                    query.user_id // User ID for the query
                );
                acorn_index->acorn.efSearch = ef_search;
                //warm up
                acorn_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data(),
                    fields_map.data()
                );
                // Perform ACORN search (timed)
                auto start = std::chrono::high_resolution_clock::now();
                acorn_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data(),
                    fields_map.data()
                );
                auto end = std::chrono::high_resolution_clock::now();
                partition_search_time = std::chrono::duration<double>(end - start).count();
            } else if (hnsw_index) {
                hnsw_index->hnsw.efSearch = ef_search;
                //warm up
                hnsw_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data()
                );
                // Perform HNSW search (timed)
                auto start = std::chrono::high_resolution_clock::now();
                hnsw_index->search(
                    1,
                    query.query_vector.data(),
                    query.topk,
                    distances.data(),
                    indices.data()
                );
                auto end = std::chrono::high_resolution_clock::now();
                partition_search_time = std::chrono::duration<double>(end - start).count();
            }

            // Add this partition's search time to total_time
            total_time += partition_search_time;

            // Add results for this partition to combined_results
            for (size_t i = 0; i < query.topk; ++i) {
                if (indices[i] >= 0 && static_cast<size_t>(indices[i]) < document_block_map.size()) {
                    combined_results.insert(
                        {distances[i], document_block_map[indices[i]]}
                    );
                }
            }


            // Release the loaded index from memory
            acorn_index.reset();
            hnsw_index.reset();
        }

        // Step 3: Ensure combined_results has top-k elements
        while (combined_results.size() < query.topk) {
            combined_results.insert({std::numeric_limits<float>::infinity(), {-1, -1}}); // Add placeholder
        }

        // Sort combined_results by `first` (distance)
        std::vector<std::pair<float, std::pair<int, int>>> sorted_results(combined_results.begin(), combined_results.end());
        std::sort(sorted_results.begin(), sorted_results.end(), [](const auto &a, const auto &b) {
            return a.first < b.first; // Sort by `first` (distance) ascending
        });

        // Convert query results into global results (only top-k)
        size_t count = 0;
        for (const auto &result : sorted_results) {
            if (count >= query.topk) break;
            all_retrieved_results.push_back(result.second);
            ++count;
        }
    }

    // Step 4: Compute recall
    avg_recall = compute_recall(ground_truth, all_retrieved_results, topk);

    // Compute average time per query
    total_time /= queries.size();

    return {total_time, avg_recall};
}
