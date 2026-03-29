#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <json_utils.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <set>
#include <pqxx/pqxx>

// Parse a vector from a string representation (e.g., "[1.0, 2.0, 3.0]")
std::vector<float> parse_vector(const std::string &vector_str);

// Compute recall given ground truth and retrieved results
double compute_recall(
    const std::vector<std::pair<int, int> > &ground_truth,
    const std::vector<std::pair<int, int> > &retrieved,
    size_t topk
);

// Fetch roles, documents, and role-to-document mappings
std::tuple<std::vector<int>, std::vector<int>, std::unordered_map<int, std::set<int> > >
fetch_role_to_documents(const std::string &conn_info);

// Compute brute-force ground truth for queries
std::vector<std::pair<int, int> > compute_groundtruth(
    const std::vector<Query> &queries,
    const std::string &conn_info
);

// Fetch metadata for filtering based on user_id
std::vector<char> build_acorn_filter_ids_map(
    const std::vector<Query> &queries,
    const std::string &conn_info
);

std::vector<char> generate_fields_map(
    const std::string &conn_info,
    const std::string &table_name,
    int user_id
);

std::unordered_map<std::string, std::vector<std::pair<int, int> > > fetch_all_document_block_maps(
    const std::string &conn_info
);

double get_table_size_in_mb(const std::string &table_name, pqxx::connection &conn);

double calculate_table_sizes(const std::vector<std::string> &tables, pqxx::connection &conn);

double calculate_index_file_sizes(const std::string &directory_path);
std::pair<double, double> print_database_and_index_statistics(const std::string &conn_info);
#endif // BENCHMARK_UTILS_H
