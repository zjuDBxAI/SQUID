#include "index_creation.h"
#include "benchmark_utils.h"
#include <pqxx/pqxx>
#include <iostream>
#include <vector>
#include <set>
#include <filesystem>
#include <unordered_map>
#include "faiss/impl/index_write.cpp"

HNSWIndexWithMetadata create_hnsw_index(
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    int d = 300;

    // Fetch all block vectors and their corresponding document IDs
    pqxx::result vector_res = txn.exec(
        "SELECT document_id, block_id, vector "
        "FROM documentblocks "
        "ORDER BY document_id, block_id;"
    );

    std::vector<std::vector<float> > all_vectors;
    std::vector<int> metadata; // Document IDs as metadata
    std::vector<std::pair<int, int> > document_block_map; // Pair of <document_id, block_id>

    for (const auto &row: vector_res) {
        all_vectors.push_back(parse_vector(row[2].c_str())); // Parse vector from "vector" column
        metadata.push_back(row[0].as<int>()); // document_id as metadata
        document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>()); // <document_id, block_id>
    }

    if (all_vectors.empty() || document_block_map.empty()) {
        throw std::runtime_error("No vectors or document-block mappings found in documentblocks.");
    }

    // Flatten the vectors
    std::vector<float> flattened_vectors;
    for (const auto &vec: all_vectors) {
        if (vec.size() != d) {
            throw std::runtime_error("Vector dimension mismatch!");
        }
        flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
    }

    // Validate the flattened data
    if (flattened_vectors.size() != all_vectors.size() * d) {
        throw std::runtime_error("Flattened vector size mismatch!");
    }

    float *xb = flattened_vectors.data();

    // Construct ACORN index with document_id metadata
    int M_hnsw = 32;
    auto index = std::make_unique<faiss::IndexHNSWFlat>(d, M_hnsw);

    // Add vectors to the index
    index->add(all_vectors.size(), xb);

    txn.commit();

    std::cout << "HNSW index created with " << all_vectors.size()
            << " block vectors and document_id metadata." << std::endl;

    return HNSWIndexWithMetadata{
        std::move(index), // ACORN index
        std::move(document_block_map) // Document-block mapping
    };
}

// Helper function to fetch all necessary data
std::tuple<std::vector<std::vector<float> >, std::vector<int>, std::vector<std::pair<int, int> > >
fetch_vectors_and_metadata(pqxx::work &txn) {
    pqxx::result vector_res = txn.exec(
        "SELECT document_id, block_id, vector "
        "FROM documentblocks "
        "ORDER BY document_id, block_id;"
    );

    std::vector<std::vector<float> > all_vectors;
    std::vector<int> metadata; // Document IDs as metadata
    std::vector<std::pair<int, int> > document_block_map; // Pair of <document_id, block_id>

    for (const auto &row: vector_res) {
        all_vectors.push_back(parse_vector(row[2].c_str())); // Parse vector from "vector" column
        metadata.push_back(row[0].as<int>()); // document_id as metadata
        document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>()); // <document_id, block_id>
    }

    if (all_vectors.empty() || document_block_map.empty()) {
        throw std::runtime_error("No vectors or document-block mappings found in documentblocks.");
    }

    return {std::move(all_vectors), std::move(metadata), std::move(document_block_map)};
}


ACORNIndexWithMetadata create_acorn_index(const std::string &conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    int d = 300, M = 32, gamma = 12, M_beta = 64;

    // Determine the index file path
    std::filesystem::path folder_path = get_index_storage_root();
    std::filesystem::create_directories(folder_path);
    std::filesystem::path index_filename = folder_path / "acorn_index.faiss";

    if (std::filesystem::exists(index_filename)) {
        std::cout << "Index file found. Loading ACORN index from file: " << index_filename << std::endl;
        std::unique_ptr<faiss::IndexACORNFlat> index(
            dynamic_cast<faiss::IndexACORNFlat *>(faiss::read_index(index_filename.c_str()))
        );

        if (!index) {
            throw std::runtime_error("Failed to load ACORN index from file.");
        }

        auto [_, __, document_block_map] = fetch_vectors_and_metadata(txn);

        std::cout << "ACORN index loaded successfully with " << document_block_map.size()
                << " document-block mappings." << std::endl;

        txn.commit();
        return ACORNIndexWithMetadata{
            std::move(index), // Loaded index
            std::move(document_block_map) // Document-block mapping
        };
    }

    std::cout << "Index file not found. Creating ACORN index..." << std::endl;

    auto [all_vectors, metadata, document_block_map] = fetch_vectors_and_metadata(txn);

    // Flatten the vectors for indexing
    std::vector<float> flattened_vectors;
    for (const auto &vec: all_vectors) {
        if (vec.size() != d) {
            throw std::runtime_error("Vector dimension mismatch!");
        }
        flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
    }

    // Validate the flattened data
    if (flattened_vectors.size() != all_vectors.size() * d) {
        throw std::runtime_error("Flattened vector size mismatch!");
    }

    float *xb = flattened_vectors.data();

    // Construct ACORN index with document_id metadata
    auto index = std::make_unique<faiss::IndexACORNFlat>(d, M, gamma, metadata, M_beta);

    // Add vectors to the ACORN index
    index->add(all_vectors.size(), xb);

    // Save the ACORN index to a file
    faiss::write_index(index.get(), index_filename.c_str());
    std::cout << "ACORN index saved to file: " << index_filename << std::endl;

    txn.commit();

    std::cout << "ACORN index created with " << all_vectors.size()
            << " block vectors and document_id metadata." << std::endl;

    return ACORNIndexWithMetadata{
        std::move(index), // Created index
        std::move(document_block_map) // Document-block mapping
    };
}


void try_create_acorn_index(const std::string &conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    int d = 300, M = 32, gamma = 12, M_beta = 64;

    // Determine the index file path
    std::filesystem::path folder_path = get_index_storage_root();
    std::filesystem::create_directories(folder_path);
    std::filesystem::path index_filename = folder_path / "acorn_index.faiss";

    if (!std::filesystem::exists(index_filename)) {
        std::cout << "Index file not found. Creating ACORN index..." << std::endl;

        auto [all_vectors, metadata, document_block_map] = fetch_vectors_and_metadata(txn);

        // Flatten the vectors for indexing
        std::vector<float> flattened_vectors;
        for (const auto &vec: all_vectors) {
            if (vec.size() != d) {
                throw std::runtime_error("Vector dimension mismatch!");
            }
            flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
        }

        // Validate the flattened data
        if (flattened_vectors.size() != all_vectors.size() * d) {
            throw std::runtime_error("Flattened vector size mismatch!");
        }

        float *xb = flattened_vectors.data();

        // Construct ACORN index with document_id metadata
        auto index = std::make_unique<faiss::IndexACORNFlat>(d, M, gamma, metadata, M_beta);

        // Add vectors to the ACORN index
        index->add(all_vectors.size(), xb);

        // Save the ACORN index to a file
        faiss::write_index(index.get(), index_filename.c_str());
        std::cout << "ACORN index saved to file: " << index_filename << std::endl;

        txn.commit();

        std::cout << "ACORN index created with " << all_vectors.size()
                << " block vectors and document_id metadata." << std::endl;
    }
}

// Create dynamic partition indices
std::unordered_map<std::string, PartitionIndex> create_dynamic_partition_indices(
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Retrieve all partition tables
    pqxx::result res = txn.exec(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'documentblocks_partition_%';"
    );
    std::vector<std::string> partition_tables;
    for (const auto &row: res) {
        partition_tables.push_back(row[0].c_str());
    }

    // Precompute role-document mapping
    auto [roles, documents, role_to_documents] = fetch_role_to_documents(conn_info);

    std::unordered_map<std::string, PartitionIndex> partition_indices;

    for (const auto &table_name: partition_tables) {
        // Define the index file path
        std::filesystem::path folder_path = get_index_storage_root() / "dynamic_partition";
        std::filesystem::path index_filename = folder_path / (table_name + ".faiss");
        std::filesystem::create_directories(folder_path);

        PartitionIndex partition_index;

        // Check if index file exists
        if (std::filesystem::exists(index_filename)) {
            std::cout << "Loading existing index for partition: " << table_name << std::endl;

            // Load the saved index
            auto loaded_index = std::unique_ptr<faiss::Index>(faiss::read_index(index_filename.c_str()));
            if (auto *acorn_index = dynamic_cast<faiss::IndexACORNFlat *>(loaded_index.get())) {
                partition_index.acorn_index = std::unique_ptr<faiss::IndexACORNFlat>(acorn_index);
                loaded_index.release(); // Release ownership
            } else if (auto *hnsw_index = dynamic_cast<faiss::IndexHNSWFlat *>(loaded_index.get())) {
                partition_index.hnsw_index = std::unique_ptr<faiss::IndexHNSWFlat>(hnsw_index);
                loaded_index.release(); // Release ownership
            } else {
                throw std::runtime_error("Unexpected index type for partition: " + table_name);
            }

            // Fetch document-block mapping
            pqxx::result doc_res = txn.exec(
                "SELECT document_id, block_id FROM " + table_name + " ORDER BY document_id, block_id"
            );
            for (const auto &row: doc_res) {
                partition_index.document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>());
            }

            partition_indices[table_name] = std::move(partition_index);
            continue;
        }

        // If no existing index, create a new one
        std::cout << "Creating new index for partition: " << table_name << std::endl;

        // Fetch document IDs, block IDs, and vectors for this partition
        pqxx::result doc_res = txn.exec(
            "SELECT document_id, block_id, vector FROM " + table_name + " ORDER BY document_id, block_id"
        );
        std::vector<std::pair<int, int> > document_block_map;
        std::vector<std::vector<float> > partition_vectors;

        for (const auto &row: doc_res) {
            document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>());
            partition_vectors.push_back(parse_vector(row[2].c_str()));
        }

        if (partition_vectors.empty()) {
            throw std::runtime_error("No vectors found in partition: " + table_name);
        }

        partition_index.document_block_map = std::move(document_block_map);

        // Flatten vectors for FAISS
        int d = 300;
        std::vector<float> flattened_vectors;
        for (const auto &vec: partition_vectors) {
            if (vec.size() != d) {
                throw std::runtime_error("Vector dimension mismatch in partition: " + table_name);
            }
            flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
        }

        // Fetch roles associated with this partition
        std::string partition_id = table_name.substr(table_name.find_last_of('_') + 1);
        pqxx::result role_res = txn.exec(
            "SELECT role_id FROM rolepartitions WHERE partition_id = " + txn.quote(partition_id)
        );
        std::set<int> associated_roles;
        for (const auto &row: role_res) {
            associated_roles.insert(row[0].as<int>());
        }

        // Determine if this partition should skip ACORN indexing
        bool skip_rls = true;
        for (const auto &role: associated_roles) {
            const auto &role_docs = role_to_documents[role];
            std::set<int> partition_documents;
            for (const auto &doc_block: partition_index.document_block_map) {
                partition_documents.insert(doc_block.first); // Extract document_id
            }
            if (!std::includes(role_docs.begin(), role_docs.end(),
                               partition_documents.begin(), partition_documents.end())) {
                skip_rls = false;
                break;
            }
        }

        if (skip_rls) {
            std::cout << "Skipping RLS for partition: " << table_name << " (Using HNSW Index)" << std::endl;

            // Create HNSW index
            int M_hnsw = 32;
            auto hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(d, M_hnsw);
            hnsw_index->add(partition_vectors.size(), flattened_vectors.data());
            partition_index.hnsw_index = std::move(hnsw_index);
        } else {
            std::cout << "Building ACORN index for partition: " << table_name << std::endl;

            // Create ACORN index
            int M = 32, gamma = 12, M_beta = 64;
            std::vector<int> metadata; // Document IDs as metadata
            for (const auto &[doc_id, _]: partition_index.document_block_map) {
                metadata.push_back(doc_id);
            }

            auto acorn_index = std::make_unique<faiss::IndexACORNFlat>(d, M, gamma, metadata, M_beta);
            acorn_index->add(partition_vectors.size(), flattened_vectors.data());
            partition_index.acorn_index = std::move(acorn_index);
        }

        // Save the index to a file
        faiss::write_index(
            partition_index.acorn_index
                ? static_cast<faiss::Index *>(partition_index.acorn_index.get())
                : static_cast<faiss::Index *>(partition_index.hnsw_index.get()),
            index_filename.c_str()
        );

        std::cout << "Index saved to file: " << index_filename << std::endl;

        partition_indices[table_name] = std::move(partition_index);
    }

    txn.commit();
    return partition_indices;
}

std::vector<int> parse_pg_array(const std::string& pg_array) {
    // Parses PostgreSQL's integer[] format into a std::vector<int>
    std::vector<int> result;
    std::stringstream ss(pg_array.substr(1, pg_array.size() - 2)); // Remove curly brackets '{}'
    std::string token;

    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

// Create dynamic partition indices
void try_create_dynamic_partition_indices(
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Retrieve all partition tables
    pqxx::result res = txn.exec(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'documentblocks_partition_%';"
    );
    std::vector<std::string> partition_tables;
    for (const auto &row: res) {
        partition_tables.push_back(row[0].c_str());
    }

    // Precompute role-document mapping
    auto [roles, documents, role_to_documents] = fetch_role_to_documents(conn_info);

    std::unordered_map<std::string, PartitionIndex> partition_indices;

    for (const auto &table_name: partition_tables) {
        // Define the index file path
        std::filesystem::path folder_path = get_index_storage_root() / "dynamic_partition";
        std::filesystem::create_directories(folder_path);
        std::filesystem::path index_filename = folder_path / (table_name + ".faiss");

        // Check whether existing index matches current table size; rebuild if mismatch
        bool needs_rebuild = true;
        std::size_t table_count = 0;
        {
            pqxx::result count_res = txn.exec("SELECT COUNT(*) FROM " + table_name + ";");
            table_count = count_res[0][0].as<std::size_t>();
        }

        if (std::filesystem::exists(index_filename)) {
            try {
                std::unique_ptr<faiss::Index> existing(faiss::read_index(index_filename.c_str()));
                if (existing && existing->ntotal == static_cast<long>(table_count)) {
                    needs_rebuild = false;
                } else {
                    std::cout << "Rebuilding index for partition: " << table_name
                              << " (size mismatch: index has "
                              << (existing ? existing->ntotal : 0)
                              << ", table has " << table_count << ")" << std::endl;
                }
            } catch (const std::exception &e) {
                std::cout << "Failed to load existing index for partition " << table_name
                          << ": " << e.what() << ". Rebuilding." << std::endl;
            }
        }

        if (!needs_rebuild) {
            continue;
        }

        if (table_count == 0) {
            std::cout << "Skipping partition " << table_name << " because it has no rows." << std::endl;
            continue;
        }

        std::cout << (std::filesystem::exists(index_filename) ? "Updating" : "Creating")
                  << " index for partition: " << table_name << std::endl;

        // Fetch document IDs, block IDs, and vectors for this partition
        pqxx::result doc_res = txn.exec(
            "SELECT document_id, block_id, vector FROM " + table_name + " ORDER BY document_id, block_id"
        );
        std::vector<std::pair<int, int> > document_block_map;
        std::vector<std::vector<float> > partition_vectors;

        document_block_map.reserve(doc_res.size());
        partition_vectors.reserve(doc_res.size());
        for (const auto &row: doc_res) {
            document_block_map.emplace_back(row[0].as<int>(), row[1].as<int>());
            partition_vectors.push_back(parse_vector(row[2].c_str()));
        }

        if (partition_vectors.empty()) {
            std::filesystem::remove(index_filename);
            throw std::runtime_error("No vectors found in partition: " + table_name);
        }

        PartitionIndex partition_index;
        partition_index.document_block_map = std::move(document_block_map);

        // Flatten vectors for FAISS
        int d = 300;
        std::vector<float> flattened_vectors;
        for (const auto &vec: partition_vectors) {
            if (vec.size() != d) {
                throw std::runtime_error("Vector dimension mismatch in partition: " + table_name);
            }
            flattened_vectors.insert(flattened_vectors.end(), vec.begin(), vec.end());
        }

        // Fetch roles associated with this partition
        std::string partition_id = table_name.substr(table_name.find_last_of('_') + 1);
        pqxx::result comb_role_res = txn.exec(
    "SELECT comb_role FROM CombRolePartitions WHERE partition_id = " + txn.quote(partition_id)
);

        // Store comb_roles as sorted sets for consistency
        std::set<std::vector<int>> associated_comb_roles;
        for (const auto &row : comb_role_res) {
            std::vector<int> comb_role = parse_pg_array(row[0].c_str());    // Extract comb_role array
            std::sort(comb_role.begin(), comb_role.end());  // Ensure sorted order
            associated_comb_roles.insert(comb_role);
        }

        // Determine if RLS can be skipped based on comb_role's document access
        bool skip_rls = true;
        for (const auto &comb_role : associated_comb_roles) {
            std::set<int> comb_docs;  // Collect all required documents for this comb_role
            for (const auto &role : comb_role) {
                const auto &role_docs = role_to_documents[role];
                comb_docs.insert(role_docs.begin(), role_docs.end());
            }

            // Check if partition contains documents outside the required comb_role scope
            std::set<int> partition_documents;
            for (const auto &doc_block : partition_index.document_block_map) {
                partition_documents.insert(doc_block.first); // Extract document_id
            }

            if (!std::includes(comb_docs.begin(), comb_docs.end(),
                               partition_documents.begin(), partition_documents.end())) {
                skip_rls = false;
                break;
                               }
        }

        if (skip_rls) {
            std::cout << "Skipping RLS for partition: " << table_name << " (Using HNSW Index)" << std::endl;

            // Create HNSW index
            int M_hnsw = 32;
            auto hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(d, M_hnsw);
            hnsw_index->add(partition_vectors.size(), flattened_vectors.data());
            partition_index.hnsw_index = std::move(hnsw_index);
        } else {
            std::cout << "Building ACORN index for partition: " << table_name << std::endl;

            // Create ACORN index
            int M = 32, gamma = 12, M_beta = 64;
            std::vector<int> metadata; // Document IDs as metadata
            for (const auto &[doc_id, _]: partition_index.document_block_map) {
                metadata.push_back(doc_id);
            }

            auto acorn_index = std::make_unique<faiss::IndexACORNFlat>(d, M, gamma, metadata, M_beta);
            acorn_index->add(partition_vectors.size(), flattened_vectors.data());
            partition_index.acorn_index = std::move(acorn_index);
        }

        // Save the index to a file
        faiss::write_index(
            partition_index.acorn_index
                ? static_cast<faiss::Index *>(partition_index.acorn_index.get())
                : static_cast<faiss::Index *>(partition_index.hnsw_index.get()),
            index_filename.c_str()
        );

        std::cout << "Index saved to file: " << index_filename << std::endl;

        partition_indices[table_name] = std::move(partition_index);
    }

    txn.commit();
}
