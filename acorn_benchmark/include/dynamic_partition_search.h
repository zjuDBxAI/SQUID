#ifndef DYNAMIC_PARTITION_SEARCH_H
#define DYNAMIC_PARTITION_SEARCH_H

#include "json_utils.h"
#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include "index_creation.h"

// Dynamic partition search with ACORN and HNSW
std::tuple<double, double> benchmark_dynamic_partition_search(
    const std::vector<Query>& queries,
    const std::unordered_map<std::string, PartitionIndex>& partition_indices,
    const std::string& conn_info
);

std::tuple<double, double> benchmark_dynamic_partition_search_with_reading_index(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    int ef_search
);

#endif // DYNAMIC_PARTITION_SEARCH_H