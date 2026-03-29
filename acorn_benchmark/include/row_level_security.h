#ifndef ROW_LEVEL_SECURITY_SEARCH_H
#define ROW_LEVEL_SECURITY_SEARCH_H

#include <tuple>
#include <vector>
#include <string>
#include <faiss/IndexACORN.h>

#include "index_creation.h"
#include "json_utils.h" // Include definition for Query

std::tuple<double, double> hnsw_search(
    const std::vector<Query> &queries,
    const std::string &conn_info,
    const HNSWIndexWithMetadata &hnsw_index_metadata,
    std::vector<char> &fields_map
);

#endif
