#ifndef ACORN_SEARCH_H
#define ACORN_SEARCH_H

#include <tuple>
#include <vector>
#include <string>
#include <faiss/IndexACORN.h>

#include "index_creation.h"
#include "json_utils.h" // Include definition for Query

// ACORN search on documentblocks
std::tuple<double, double> acorn_search(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    const ACORNIndexWithMetadata &acorn_index_metadata,
    std::vector<char> &fields_map
);

std::tuple<double, double> acorn_search_with_reading_index(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    std::vector<char> &fields_map, // fields_map provided as a parameter
    int ef_search
);
#endif // ACORN_SEARCH_H
