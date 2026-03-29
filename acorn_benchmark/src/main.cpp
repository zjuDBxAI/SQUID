#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include "json_utils.h"
#include "acorn_search.h"
#include "benchmark_utils.h"
#include "dynamic_partition_search.h"
#include "index_creation.h"
#include "row_level_security.h"

int main() {
    try {
        // Get the root directory of the project
        std::string project_root = get_project_root();

        // Paths to the configuration and query dataset files
        std::string config_path = project_root + "/config.json";
        std::string query_dataset_path = project_root + "/basic_benchmark/query_dataset.json";

        // Parse database configuration and load queries
        std::string conn_info = parse_db_config(config_path);
        auto queries = read_queries(query_dataset_path);

        // Fetch metadata for the queries
        auto filter_ids_map = build_acorn_filter_ids_map(queries, conn_info);

        //create index
        {
            try_create_acorn_index(conn_info);
            try_create_dynamic_partition_indices(conn_info);
        }

        // Perform ACORN search
        omp_set_num_threads(1);
        std::vector<int> ef_search_acorn_values = {7,8,9,10,11,12,13,15}; // Array of ACORN ef_search values
        std::vector<int> ef_search_partition_values = {10,15,20,25,30,35,40,45,50}; // Array of Dynamic Partition ef_search values

        bool test_acorn = true;
        bool test_dynamic_partition =true;
        // Loop over ACORN ef_search values
        if (test_acorn == true) {
            for (int ef_search_acorn: ef_search_acorn_values) {
                auto [acorn_time, acorn_recall] =
                        acorn_search_with_reading_index(queries, conn_info, filter_ids_map, ef_search_acorn);
                std::cout << "ACORN Search (ef_search=" << ef_search_acorn << ") - Time: " << acorn_time
                        << "s, Recall: " << acorn_recall << std::endl;

                // Calculate space usage for ACORN
                auto [acorn_space, _] = print_database_and_index_statistics(conn_info);

                // Prepare JSON data
                nlohmann::json acorn_data = {
                    {"search_time", acorn_time},
                    {"recall", acorn_recall},
                    {"space", acorn_space}
                };

                // Write JSON data to file
                std::string output_dir = project_root + "/acorn_benchmark/outputdata";
                std::filesystem::create_directories(output_dir);

                std::ofstream acorn_file(output_dir + "/acorn_" + std::to_string(ef_search_acorn) + ".json");
                acorn_file << acorn_data.dump(4);
                acorn_file.close();
            }
        }
        // Loop over Dynamic Partition ef_search values
        if (test_dynamic_partition == true) {
            for (int ef_search_partition: ef_search_partition_values) {
                auto [partition_time, partition_recall] =
                        benchmark_dynamic_partition_search_with_reading_index(queries, conn_info, ef_search_partition);
                std::cout << "Dynamic Partition Search (ef_search=" << ef_search_partition << ") - Time: " <<
                        partition_time
                        << "s, Recall: " << partition_recall << std::endl;

                // Calculate space usage for Dynamic Partition
                auto [_, partition_space] = print_database_and_index_statistics(conn_info);

                // Prepare JSON data
                nlohmann::json partition_data = {
                    {"search_time", partition_time},
                    {"recall", partition_recall},
                    {"space", partition_space}
                };

                // Write JSON data to file
                std::string output_dir = project_root + "/acorn_benchmark/outputdata";
                std::filesystem::create_directories(output_dir);

                std::ofstream partition_file(
                    output_dir + "/dynamic_partition_" + std::to_string(ef_search_partition) + ".json");
                partition_file << partition_data.dump(4);
                partition_file.close();
            }
        }


        std::cout << "All statistics written to JSON files in " << project_root + "/acorn_benchmark/outputdata" <<
                std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
