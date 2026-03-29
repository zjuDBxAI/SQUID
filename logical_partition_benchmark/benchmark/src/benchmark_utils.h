#ifndef LOGICAL_PARTITION_BENCHMARK_SRC_BENCHMARK_UTILS_H
#define LOGICAL_PARTITION_BENCHMARK_SRC_BENCHMARK_UTILS_H

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <pqxx/pqxx>
#include <faiss/impl/HNSW.h>

#include "shared_vector_table.h"

namespace benchmark_utils {

using pointer_benchmark::SharedVectorTable;
using DocBlockId = SharedVectorTable::DocBlockId;
using DocBlockIdHash = SharedVectorTable::DocBlockIdHash;
using RoleSet = std::unordered_set<int>;
using json = nlohmann::json;

struct Query {
    int user_id = 0;
    std::vector<float> query_vector;
    int topk = 10;
};

struct HNSWGraphStats {
    size_t nodes = 0;
    size_t neighbor_bytes = 0;
    size_t offsets_bytes = 0;
    size_t levels_bytes = 0;
    size_t assign_probas_bytes = 0;
    size_t cum_nneighbor_bytes = 0;

    size_t total_bytes() const {
        return neighbor_bytes + offsets_bytes + levels_bytes + assign_probas_bytes + cum_nneighbor_bytes;
    }
};

inline void normalize_vector(std::vector<float>& vec) {
    float norm_sq = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f);
    if (norm_sq <= 0.0f) {
        return;
    }
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    for (auto& value : vec) {
        value *= inv_norm;
    }
}

inline std::filesystem::path project_root() {
    auto path = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i) {
        // Look for project root indicators: config.json AND basic_benchmark directory
        if (std::filesystem::exists(path / "config.json") &&
            std::filesystem::exists(path / "basic_benchmark")) {
            return path;
        }
        if (!path.has_parent_path()) {
            break;
        }
        path = path.parent_path();
    }
    return std::filesystem::current_path();
}

inline std::string load_connection_string(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file at " + config_path.string());
    }

    json cfg;
    file >> cfg;

    const std::string host = cfg.value("host", "localhost");
    std::string port;
    if (cfg.contains("port")) {
        if (cfg["port"].is_string()) {
            port = cfg["port"].get<std::string>();
        } else {
            port = std::to_string(cfg["port"].get<int>());
        }
    } else {
        port = "5432";
    }
    const std::string dbname = cfg.contains("dbname") ? cfg["dbname"].get<std::string>()
                                   : cfg.value("database", std::string("your_db"));
    const std::string user = cfg.value("user", std::string("your_user"));
    const std::string password = cfg.value("password", std::string("your_password"));

    return "host=" + host + " port=" + port + " dbname=" + dbname +
           " user=" + user + " password=" + password;
}

inline std::filesystem::path load_index_storage_path() {
    const auto root = project_root();
    const std::filesystem::path benchmark_config =
        root / "logical_partition_benchmark" / "benchmark" / "config.json";
    const std::filesystem::path current_config = std::filesystem::current_path() / "config.json";
    const std::filesystem::path root_config = root / "config.json";

    const std::vector<std::filesystem::path> configs = {
        benchmark_config,
        current_config,
        root_config
    };

    for (const auto& config_path : configs) {
        if (!std::filesystem::exists(config_path)) {
            continue;
        }

        std::ifstream file(config_path);
        if (!file.is_open()) {
            continue;
        }

        json cfg;
        try {
            file >> cfg;
        } catch (...) {
            continue;
        }

        if (!cfg.contains("index_storage_path")) {
            continue;
        }

        std::string path_str = cfg["index_storage_path"].get<std::string>();
        std::filesystem::path resolved(path_str);
        if (!resolved.is_absolute()) {
            resolved = std::filesystem::absolute(config_path.parent_path() / resolved);
        }
        std::filesystem::create_directories(resolved);
        return resolved;
    }

    // Fallback: use default path
    const std::filesystem::path default_path = "/pgsql_data";
    std::filesystem::create_directories(default_path);
    return default_path;
}

struct HNSWConfig {
    int M = 16;
    int ef_construction = 64;
};

inline HNSWConfig load_hnsw_config() {
    HNSWConfig config;

    // Try to find hnsw_config.json in the dynamic_logical_partition directory
    auto project_root_path = project_root();
    auto hnsw_config_path = project_root_path / "logical_partition_benchmark" / "dynamic_logical_partition" / "hnsw_config.json";

    if (std::filesystem::exists(hnsw_config_path)) {
        std::ifstream file(hnsw_config_path);
        if (file.is_open()) {
            json cfg;
            file >> cfg;
            if (cfg.contains("M")) {
                config.M = cfg["M"].get<int>();
            }
            if (cfg.contains("ef_construction")) {
                config.ef_construction = cfg["ef_construction"].get<int>();
            }
            file.close();
            return config;
        }
    }

    // Fallback: use default values (pgvector defaults)
    return config;
}

inline std::vector<Query> load_queries(const std::filesystem::path& path,
                                       bool normalize = false) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open query dataset at " + path.string());
    }

    json j;
    file >> j;

    std::vector<Query> queries;
    queries.reserve(j.size());

    for (const auto& item : j) {
        Query q;
        q.user_id = item.at("user_id").get<int>();
        if (item.at("query_vector").is_array()) {
            q.query_vector = item.at("query_vector").get<std::vector<float>>();
        } else {
            std::string vec = item.at("query_vector").get<std::string>();
            if (!vec.empty() && vec.front() == '[') vec.erase(vec.begin());
            if (!vec.empty() && vec.back() == ']') vec.pop_back();
            std::stringstream ss(vec);
            std::string token;
            while (std::getline(ss, token, ',')) {
                q.query_vector.push_back(std::stof(token));
            }
        }
        if (item.contains("topk")) {
            q.topk = item.at("topk").get<int>();
        }
        if (normalize) {
            normalize_vector(q.query_vector);
        }
        queries.push_back(std::move(q));
    }

    return queries;
}

inline double percentile(std::vector<double>& data, double p) {
    if (data.empty()) {
        return 0.0;
    }
    std::sort(data.begin(), data.end());
    size_t idx = static_cast<size_t>(p * static_cast<double>(data.size()));
    if (idx >= data.size()) {
        idx = data.size() - 1;
    }
    return data[idx];
}

inline HNSWGraphStats compute_hnsw_graph_stats(const faiss::HNSW& hnsw) {
    HNSWGraphStats stats;
    stats.nodes = hnsw.levels.size();
    stats.neighbor_bytes = hnsw.neighbors.byte_size();
    stats.offsets_bytes = hnsw.offsets.size() * sizeof(size_t);
    stats.levels_bytes = hnsw.levels.size() * sizeof(int);
    stats.assign_probas_bytes = hnsw.assign_probas.size() * sizeof(double);
    stats.cum_nneighbor_bytes = hnsw.cum_nneighbor_per_level.size() * sizeof(int);
    return stats;
}

inline std::shared_ptr<SharedVectorTable> load_shared_table(int dimension,
                                                            const std::filesystem::path& cache_path,
                                                            const std::string& conn_info,
                                                            faiss::MetricType metric = faiss::METRIC_L2,
                                                            bool normalize = false) {
    auto table = std::make_shared<SharedVectorTable>(dimension, metric, normalize);
    if (!cache_path.empty() && std::filesystem::exists(cache_path) &&
        table->load_vectors(cache_path.string())) {
        return table;
    }

    table->load_from_database(conn_info);
    if (!cache_path.empty()) {
        std::filesystem::create_directories(cache_path.parent_path());
        table->save_vectors(cache_path.string());
    }
    return table;
}

inline std::vector<std::string> fetch_tables_with_prefix(const std::string& conn_info,
                                                         const std::string& prefix) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);
    auto res = txn.exec("SELECT tablename FROM pg_tables WHERE tablename LIKE " +
                        txn.quote(prefix + "%") + " ORDER BY tablename");
    std::vector<std::string> tables;
    tables.reserve(res.size());
    for (const auto& row : res) {
        tables.push_back(row[0].as<std::string>());
    }
    txn.commit();
    return tables;
}

inline RoleSet to_roleset(const std::vector<int>& roles) {
    return RoleSet(roles.begin(), roles.end());
}

inline std::unordered_map<int, RoleSet> load_user_roles(const std::string& conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);
    auto res = txn.exec("SELECT user_id, role_id FROM UserRoles");
    std::unordered_map<int, RoleSet> user_roles;
    for (const auto& row : res) {
        user_roles[row[0].as<int>()].insert(row[1].as<int>());
    }
    txn.commit();
    return user_roles;
}

inline std::unordered_map<int, RoleSet> load_document_roles(const std::string& conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    auto tables = txn.exec("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_role_%'");
    std::unordered_map<int, RoleSet> doc_roles;

    for (const auto& table_row : tables) {
        const std::string table = table_row[0].as<std::string>();
        const auto pos = table.find_last_of('_');
        if (pos == std::string::npos) continue;
        const int role_id = std::stoi(table.substr(pos + 1));
        auto docs = txn.exec("SELECT DISTINCT document_id FROM " + table);
        for (const auto& doc_row : docs) {
            doc_roles[doc_row[0].as<int>()].insert(role_id);
        }
    }

    txn.commit();
    return doc_roles;
}

inline std::unordered_map<DocBlockId, std::vector<int>, DocBlockIdHash>
load_doc_block_roles(const std::string& conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    auto tables = txn.exec("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_role_%'");
    std::unordered_map<DocBlockId, std::vector<int>, DocBlockIdHash> doc_roles_map;

    for (const auto& table_row : tables) {
        const std::string table = table_row[0].as<std::string>();
        const auto pos = table.find_last_of('_');
        if (pos == std::string::npos) continue;
        const int role_id = std::stoi(table.substr(pos + 1));

        auto docs = txn.exec("SELECT document_id, block_id FROM " + table);
        for (const auto& doc_row : docs) {
            DocBlockId key{doc_row[0].as<int>(), doc_row[1].as<int>()};
            auto& roles = doc_roles_map[key];
            if (std::find(roles.begin(), roles.end(), role_id) == roles.end()) {
                roles.push_back(role_id);
            }
        }
    }

    txn.commit();
    return doc_roles_map;
}

inline std::vector<int> parse_pg_int_array(const std::string& text) {
    std::vector<int> values;
    if (text.size() < 2) {
        return values;
    }
    auto body = text.substr(1, text.size() - 2);
    std::stringstream ss(body);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

inline std::filesystem::path ground_truth_cache_path() {
    return project_root() / "basic_benchmark" / "ground_truth_cache.json";
}

inline bool load_ground_truth_from_cache(
    const std::filesystem::path& path,
    size_t expected_queries,
    std::vector<std::vector<DocBlockId>>& out,
    const std::unordered_map<int, RoleSet>& doc_roles) {
    std::ifstream gt_file(path);
    if (!gt_file.good()) {
        return false;
    }

    json gt_json;
    try {
        gt_file >> gt_json;
    } catch (...) {
        return false;
    }

    if (!gt_json.is_array()) {
        return false;
    }

    out.clear();
    out.reserve(gt_json.size());
    for (const auto& gt_query : gt_json) {
        if (!gt_query.is_array()) {
            return false;
        }
        std::vector<DocBlockId> results;
        results.reserve(gt_query.size());
        for (const auto& result : gt_query) {
            if (!result.is_array() || result.size() < 2) {
                return false;
            }
            int first = result[0].get<int>();
            int second = result[1].get<int>();
            const bool first_is_doc = !doc_roles.empty() && doc_roles.find(first) != doc_roles.end();
            const bool second_is_doc = !doc_roles.empty() && doc_roles.find(second) != doc_roles.end();
            if (!first_is_doc && second_is_doc) {
                std::swap(first, second);
            }
            results.emplace_back(first, second);
        }
        out.push_back(std::move(results));
    }

    if (out.size() != expected_queries) {
        out.clear();
        return false;
    }

    return true;
}

inline void save_ground_truth_to_cache(const std::filesystem::path& path,
                                       const std::vector<std::vector<DocBlockId>>& ground_truth) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    json gt_json = json::array();
    for (const auto& gt_query : ground_truth) {
        json gt_query_json = json::array();
        for (const auto& [doc_id, block_id] : gt_query) {
            gt_query_json.push_back({doc_id, block_id});
        }
        gt_json.push_back(std::move(gt_query_json));
    }

    std::ofstream out(path);
    if (out.is_open()) {
        out << gt_json.dump(2);
    }
}

inline std::vector<std::vector<DocBlockId>> compute_ground_truth(
    const std::vector<Query>& queries,
    const std::shared_ptr<SharedVectorTable>& shared_table,
    const std::unordered_map<int, RoleSet>& user_roles,
    const std::unordered_map<int, RoleSet>& doc_roles) {
    std::vector<std::vector<DocBlockId>> ground_truth;
    if (!shared_table) {
        return ground_truth;
    }

    auto* index = shared_table->get_index_flat();

    const int ntotal = static_cast<int>(index->ntotal);
    const int search_k = std::min(1000, std::max(ntotal, 0));
    std::vector<float> distances(std::max(search_k, 1));
    std::vector<faiss::idx_t> labels(std::max(search_k, 1));

    ground_truth.reserve(queries.size());

    for (const auto& query : queries) {
        auto it = user_roles.find(query.user_id);
        if (it == user_roles.end() || it->second.empty() || ntotal == 0) {
            ground_truth.emplace_back();
            continue;
        }

        const auto& allowed_roles = it->second;
        const int actual_k = std::min(search_k, ntotal);
        if (actual_k <= 0) {
            ground_truth.emplace_back();
            continue;
        }
        index->search(1, query.query_vector.data(), actual_k, distances.data(), labels.data());

        std::vector<DocBlockId> filtered;
        filtered.reserve(query.topk);

        for (int i = 0; i < actual_k && static_cast<int>(filtered.size()) < query.topk; ++i) {
            const faiss::idx_t label = labels[i];
            if (label < 0) continue;
            DocBlockId doc_block = shared_table->get_doc_block_id(label);
            auto doc_it = doc_roles.find(doc_block.first);
            if (doc_it == doc_roles.end()) continue;
            const auto& doc_role_set = doc_it->second;
            bool allowed = false;
            for (int role : doc_role_set) {
                if (allowed_roles.count(role) > 0) {
                    allowed = true;
                    break;
                }
            }
            if (allowed) {
                filtered.push_back(doc_block);
            }
        }

        ground_truth.push_back(std::move(filtered));
    }

    return ground_truth;
}

inline std::vector<std::vector<DocBlockId>> load_or_compute_ground_truth(
    const std::vector<Query>& queries,
    const std::shared_ptr<SharedVectorTable>& shared_table,
    const std::unordered_map<int, RoleSet>& user_roles,
    const std::unordered_map<int, RoleSet>& doc_roles,
    bool* loaded_from_cache = nullptr) {
    std::vector<std::vector<DocBlockId>> ground_truth;
    const auto cache_file = ground_truth_cache_path();

    bool from_cache = load_ground_truth_from_cache(cache_file, queries.size(), ground_truth, doc_roles);
    if (!from_cache) {
        ground_truth = compute_ground_truth(queries, shared_table, user_roles, doc_roles);
        if (ground_truth.size() == queries.size()) {
            save_ground_truth_to_cache(cache_file, ground_truth);
        }
    }

    // Detect pointer-style caches that store entries as [block_id, document_id]
    // and swap them back to (document_id, block_id).
    for (auto& gt_list : ground_truth) {
        if (gt_list.empty()) {
            continue;
        }
        const auto& pair = gt_list.front();
        bool first_is_doc = doc_roles.find(pair.first) != doc_roles.end();
        bool second_is_doc = doc_roles.find(pair.second) != doc_roles.end();
        if (!first_is_doc && second_is_doc) {
            for (auto& entry : gt_list) {
                std::swap(entry.first, entry.second);
            }
        }
        break;
    }

    if (loaded_from_cache) {
        *loaded_from_cache = from_cache;
    }

    return ground_truth;
}

inline double calculate_recall(const std::vector<DocBlockId>& predicted,
                               const std::vector<DocBlockId>& ground_truth) {
    if (ground_truth.empty()) {
        return 0.0;
    }
    std::unordered_set<DocBlockId, DocBlockIdHash> predicted_set(predicted.begin(), predicted.end());
    size_t hits = 0;
    for (const auto& item : ground_truth) {
        if (predicted_set.count(item) > 0) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(ground_truth.size());
}

} // namespace benchmark_utils

#endif // LOGICAL_PARTITION_BENCHMARK_SRC_BENCHMARK_UTILS_H
