#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <filesystem>
#include <string>
#include <vector>
#include <tuple>

// Query structure
struct Query {
    int user_id;
    std::vector<float> query_vector;
    int topk;
    double query_block_selectivity;
};

// Function to read queries from JSON file
std::string get_project_root();
std::vector<Query> read_queries(const std::string& filepath);
std::string parse_db_config(const std::string& config_path);
std::filesystem::path get_index_storage_root();
#endif
