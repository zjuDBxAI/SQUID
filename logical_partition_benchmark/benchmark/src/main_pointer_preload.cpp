#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <pqxx/pqxx>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>

#include "benchmark_utils.h"
#include "pointer_hnsw_index.h"
#include "shared_vector_table.h"

using pointer_benchmark::PointerHNSWIndex;
using pointer_benchmark::SharedVectorTable;
using benchmark_utils::Query;
using benchmark_utils::calculate_recall;
using benchmark_utils::compute_ground_truth;
using benchmark_utils::fetch_tables_with_prefix;
using benchmark_utils::load_document_roles;
using benchmark_utils::load_queries;
using benchmark_utils::load_shared_table;
using benchmark_utils::load_user_roles;
using benchmark_utils::percentile;
using json = nlohmann::json;

namespace {

struct Config {
    std::string db_host = "localhost";
    std::string db_port = "5432";
    std::string db_name = "your_db";
    std::string db_user = "your_user";
    std::string db_password = "your_password";

    int M = 32;
    int efConstruction = 200;
    int efSearch = 50;
    int topk = 10;
    std::string save_index_dir;
};

Config load_config(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file at " + path.string());
    }

    json j;
    file >> j;

    Config cfg;
    if (j.contains("host")) cfg.db_host = j["host"];
    if (j.contains("port")) {
        if (j["port"].is_string()) {
            cfg.db_port = j["port"].get<std::string>();
        } else {
            cfg.db_port = std::to_string(j["port"].get<int>());
        }
    }
    if (j.contains("dbname")) cfg.db_name = j["dbname"];
    else if (j.contains("database")) cfg.db_name = j["database"];
    if (j.contains("user")) cfg.db_user = j["user"];
    if (j.contains("password")) cfg.db_password = j["password"];
    if (j.contains("save_index_dir")) cfg.save_index_dir = j["save_index_dir"];
    if (j.contains("M")) cfg.M = j["M"].get<int>();
    if (j.contains("efSearch")) cfg.efSearch = j["efSearch"].get<int>();
    if (j.contains("efConstruction")) cfg.efConstruction = j["efConstruction"].get<int>();
    if (j.contains("topk")) cfg.topk = j["topk"].get<int>();
    return cfg;
}

std::string connection_string(const Config& cfg) {
    return "host=" + cfg.db_host +
           " port=" + cfg.db_port +
           " dbname=" + cfg.db_name +
           " user=" + cfg.db_user +
           " password=" + cfg.db_password;
}

class PointerPartitionCache {
public:
    explicit PointerPartitionCache(std::shared_ptr<SharedVectorTable> shared_table,
                                   int M,
                                   int efConstruction,
                                   int efSearch)
        : shared_table_(std::move(shared_table)),
          M_(M),
          efConstruction_(efConstruction),
          efSearch_(efSearch) {}

    void preload(const std::vector<std::string>& partitions,
                 const std::string& conn_info,
                 const std::filesystem::path& base_dir) {
        entries_.clear();
        name_to_index_.clear();

        std::filesystem::create_directories(base_dir);

        size_t idx = 0;
        for (const auto& partition : partitions) {
            auto pointer_idx = std::make_unique<PointerHNSWIndex>(shared_table_, M_, efConstruction_);
            std::filesystem::path graph_path = base_dir / (partition + "_graph.bin");
            if (!pointer_idx->load_graph(graph_path.string())) {
                std::cout << "[PointerCache] Graph missing for " << partition << ", rebuilding..." << std::endl;
                pointer_idx->build_from_partition(conn_info, partition);
                pointer_idx->save_graph(graph_path.string());
            }
            pointer_idx->set_ef_search(efSearch_);

            entries_.push_back({partition, std::move(pointer_idx)});
            name_to_index_[partition] = idx++;
        }

        std::cout << "[PointerCache] Preloaded " << entries_.size()
                  << " pointer partitions into memory." << std::endl;
    }

    PointerHNSWIndex* get(const std::string& partition) const {
        auto it = name_to_index_.find(partition);
        if (it == name_to_index_.end()) {
            return nullptr;
        }
        return entries_[it->second].index.get();
    }

private:
    struct Entry {
        std::string name;
        std::unique_ptr<PointerHNSWIndex> index;
    };

    std::shared_ptr<SharedVectorTable> shared_table_;
    int M_;
    int efConstruction_;
    int efSearch_;
    std::vector<Entry> entries_;
    std::unordered_map<std::string, size_t> name_to_index_;
};

} // namespace

int main() {
    try {
        const std::filesystem::path project_root =
            std::filesystem::current_path().parent_path().parent_path().parent_path();
        Config cfg = load_config(project_root / "config.json");

        auto index_storage = benchmark_utils::load_index_storage_path();
        if (cfg.save_index_dir.empty()) {
            cfg.save_index_dir = index_storage.string();
        }

        const std::string conn_info = connection_string(cfg);

        // Load queries
        const auto query_path = project_root / "basic_benchmark/query_dataset.json";
        auto queries = load_queries(query_path);
        if (queries.empty()) {
            throw std::runtime_error("No queries found in " + query_path.string());
        }
        std::cout << "Loaded " << queries.size() << " queries from \"" << query_path << "\"\n" << std::endl;

        const int dimension = static_cast<int>(queries[0].query_vector.size());

        // Shared vector table
        std::filesystem::path base_dir(cfg.save_index_dir);
        std::filesystem::create_directories(base_dir);
        auto pointer_dir = base_dir / "pointer_shared";
        std::filesystem::create_directories(pointer_dir);
        auto shared_vectors_path = pointer_dir / "shared_vectors.bin";
        auto shared_table = load_shared_table(dimension, shared_vectors_path, conn_info);
        std::cout << "Shared table has " << shared_table->size() << " vectors" << std::endl;

        // Metadata
        auto partitions = fetch_tables_with_prefix(conn_info, "documentblocks_role_");
        auto user_roles = load_user_roles(conn_info);
        auto doc_roles = load_document_roles(conn_info);

        // Ground truth (with caching)
        std::string gt_cache_file = (project_root / "pointer_benchmark/ground_truth_cache.json").string();
        std::vector<std::vector<std::pair<int, int>>> ground_truth;

        auto save_ground_truth = [&](const auto& gt) {
            json gt_json = json::array();
            for (const auto& gt_query : gt) {
                json q = json::array();
                for (const auto& [doc, block] : gt_query) {
                    q.push_back({doc, block});
                }
                gt_json.push_back(q);
            }
            std::ofstream out(gt_cache_file);
            out << gt_json.dump(2);
        };

        std::ifstream gt_file(gt_cache_file);
        if (gt_file.good()) {
            json gt_json;
            gt_file >> gt_json;
            gt_file.close();
            for (const auto& q : gt_json) {
                std::vector<std::pair<int, int>> items;
                for (const auto& entry : q) {
                    items.emplace_back(entry[0].get<int>(), entry[1].get<int>());
                }
                ground_truth.push_back(std::move(items));
            }
            std::cout << "Loaded ground truth for " << ground_truth.size() << " queries from cache" << std::endl;
        }

        if (ground_truth.size() != queries.size()) {
            std::cout << "Computing ground truth..." << std::endl;
            ground_truth = compute_ground_truth(queries, shared_table, user_roles, doc_roles);
            save_ground_truth(ground_truth);
            std::cout << "Ground truth cached to " << gt_cache_file << std::endl;
        }

        // Preload pointer graphs
        PointerPartitionCache cache(shared_table, cfg.M, cfg.efConstruction, cfg.efSearch);
        cache.preload(partitions, conn_info, pointer_dir);

        // Optionally warm up once per partition
        {
            std::vector<float> dummy(dimension, 0.0f);
            for (int i = 0; i < std::min<int>(dimension, 10); ++i) {
                dummy[i] = 1.0f;
            }
            for (const auto& partition : partitions) {
                if (auto* idx = cache.get(partition)) {
                    idx->search(dummy.data(), 1);
                }
            }
        }

        std::vector<double> query_times;
        std::vector<double> recalls;
        query_times.reserve(queries.size());
        recalls.reserve(queries.size());
        std::vector<long long> all_cycles, all_instr, all_cache_ref, all_cache_miss, all_l1d_loads, all_l1d_miss, all_llc_loads, all_llc_miss;
        std::vector<long long> all_stall_fe, all_stall_be, all_br_miss, all_ctx_sw, all_cpu_mig, all_page_fault;

        for (size_t qi = 0; qi < queries.size(); ++qi) {
            const auto& query = queries[qi];
            auto user_it = user_roles.find(query.user_id);
            if (user_it == user_roles.end() || user_it->second.empty()) {
                query_times.push_back(0.0);
                recalls.push_back(0.0);
                continue;
            }

            std::vector<std::pair<int, int>> aggregated;
            aggregated.reserve(static_cast<size_t>(query.topk) * user_it->second.size());

            // Warm-up pass (mirrors pointer baseline behaviour)
            for (int role_id : user_it->second) {
                std::string partition = "documentblocks_role_" + std::to_string(role_id);
                if (auto* index = cache.get(partition)) {
                    index->search(query.query_vector.data(), cfg.topk);
                }
            }

            // Setup perf counters
            struct perf_event_attr pe;
            memset(&pe, 0, sizeof(pe));
            pe.size = sizeof(pe); pe.disabled = 1; pe.exclude_kernel = 1; pe.exclude_hv = 1;
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CPU_CYCLES; int fd_cycles = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_INSTRUCTIONS; int fd_instr = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_CACHE_REFERENCES; int fd_cref = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_CACHE_MISSES; int fd_cmiss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND; int fd_stall_fe = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND; int fd_stall_be = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_HW_BRANCH_MISSES; int fd_br_miss = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.type = PERF_TYPE_SOFTWARE;
            pe.config = PERF_COUNT_SW_CONTEXT_SWITCHES; int fd_ctx_sw = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_SW_CPU_MIGRATIONS; int fd_cpu_mig = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = PERF_COUNT_SW_PAGE_FAULTS; int fd_page_fault = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.type = PERF_TYPE_HW_CACHE;
            pe.config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS<<16); int fd_l1d = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16); int fd_l1dm = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS<<16); int fd_llc = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            pe.config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16); int fd_llcm = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
            int fds[] = {fd_cycles, fd_instr, fd_cref, fd_cmiss, fd_l1d, fd_l1dm, fd_llc, fd_llcm, fd_stall_fe, fd_stall_be, fd_br_miss, fd_ctx_sw, fd_cpu_mig, fd_page_fault};

            long long v[14] = {0};
            const auto start = std::chrono::high_resolution_clock::now();
            for (int role_id : user_it->second) {
                std::string partition = "documentblocks_role_" + std::to_string(role_id);
                if (auto* index = cache.get(partition)) {
                    for (int fd : fds) if (fd >= 0) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }
                    auto results = index->search(query.query_vector.data(), cfg.topk);
                    for (int fd : fds) if (fd >= 0) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
                    for (int j = 0; j < 14; j++) if (fds[j] >= 0) { long long tmp; read(fds[j], &tmp, sizeof(tmp)); v[j] += tmp; }
                    aggregated.insert(aggregated.end(), results.begin(), results.end());
                }
            }
            const auto end = std::chrono::high_resolution_clock::now();

            // Close fds
            for (int fd : fds) if (fd >= 0) close(fd);
            all_cycles.push_back(v[0]); all_instr.push_back(v[1]); all_cache_ref.push_back(v[2]); all_cache_miss.push_back(v[3]);
            all_l1d_loads.push_back(v[4]); all_l1d_miss.push_back(v[5]); all_llc_loads.push_back(v[6]); all_llc_miss.push_back(v[7]);
            all_stall_fe.push_back(v[8]); all_stall_be.push_back(v[9]); all_br_miss.push_back(v[10]);
            all_ctx_sw.push_back(v[11]); all_cpu_mig.push_back(v[12]); all_page_fault.push_back(v[13]);

            if (qi == 0) {
                std::cout << "\n=== Perf Stats (First Query) ===" << std::endl;
                std::cout << "Cycles: " << v[0] << std::endl;
                std::cout << "Instructions: " << v[1] << std::endl;
                std::cout << "IPC: " << (v[0] > 0 ? (double)v[1]/v[0] : 0.0) << std::endl;
                std::cout << "Stalled cycles (frontend): " << v[8] << " (" << (v[0] > 0 ? (double)v[8]*100.0/v[0] : 0.0) << "%)" << std::endl;
                std::cout << "Stalled cycles (backend): " << v[9] << " (" << (v[0] > 0 ? (double)v[9]*100.0/v[0] : 0.0) << "%)" << std::endl;
                std::cout << "Branch misses: " << v[10] << std::endl;
                std::cout << "Context switches: " << v[11] << std::endl;
                std::cout << "CPU migrations: " << v[12] << std::endl;
                std::cout << "Page faults: " << v[13] << std::endl;
                std::cout << "Cache refs: " << v[2] << ", misses: " << v[3]
                          << " (" << (v[2] > 0 ? (double)v[3]*100.0/v[2] : 0.0) << "%)" << std::endl;
                std::cout << "L1D loads: " << v[4] << ", misses: " << v[5]
                          << " (" << (v[4] > 0 ? (double)v[5]*100.0/v[4] : 0.0) << "%)" << std::endl;
                std::cout << "LLC loads: " << v[6] << ", misses: " << v[7]
                          << " (" << (v[6] > 0 ? (double)v[7]*100.0/v[6] : 0.0) << "%)" << std::endl;
                std::cout << "================================" << std::endl;
            }
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            query_times.push_back(elapsed_ms);

            const auto& gt = (qi < ground_truth.size()) ? ground_truth[qi]
                                                        : std::vector<std::pair<int, int>>{};
            recalls.push_back(calculate_recall(aggregated, gt));
        }

        double avg_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) /
                          static_cast<double>(query_times.size());
        double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) /
                            static_cast<double>(recalls.size());

        auto times_copy = query_times;
        double p50 = percentile(times_copy, 0.5);
        double p90 = percentile(times_copy, 0.9);
        double p95 = percentile(times_copy, 0.95);
        double p99 = percentile(times_copy, 0.99);

        std::cout << "\n========================================" << std::endl;
        std::cout << "SCENARIO: Pointer HNSW (Preloaded Graphs)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Avg time: " << avg_time << " ms" << std::endl;
        std::cout << "  P50: " << p50 << " ms, P90: " << p90
                  << " ms, P95: " << p95 << " ms, P99: " << p99 << " ms" << std::endl;
        std::cout << "  Avg recall: " << avg_recall << std::endl;

        // Print average perf stats
        auto avg_fn = [](const std::vector<long long>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0LL) / (double)v.size(); };
        double avg_cycles = avg_fn(all_cycles), avg_instr = avg_fn(all_instr), avg_cache_ref = avg_fn(all_cache_ref);
        double avg_cache_miss = avg_fn(all_cache_miss), avg_l1d_loads = avg_fn(all_l1d_loads), avg_l1d_miss = avg_fn(all_l1d_miss);
        double avg_llc_loads = avg_fn(all_llc_loads), avg_llc_miss = avg_fn(all_llc_miss);
        double avg_stall_fe = avg_fn(all_stall_fe), avg_stall_be = avg_fn(all_stall_be), avg_br_miss = avg_fn(all_br_miss);
        double avg_ctx_sw = avg_fn(all_ctx_sw), avg_cpu_mig = avg_fn(all_cpu_mig), avg_page_fault = avg_fn(all_page_fault);
        std::cout << "\n=== Average Perf Stats (All Queries) ===" << std::endl;
        std::cout << "Avg Cycles: " << avg_cycles << std::endl;
        std::cout << "Avg Instructions: " << avg_instr << std::endl;
        std::cout << "Avg IPC: " << (avg_cycles > 0 ? avg_instr/avg_cycles : 0.0) << std::endl;
        std::cout << "Avg Stalled cycles (frontend): " << avg_stall_fe << " (" << (avg_cycles > 0 ? avg_stall_fe*100.0/avg_cycles : 0.0) << "%)" << std::endl;
        std::cout << "Avg Stalled cycles (backend): " << avg_stall_be << " (" << (avg_cycles > 0 ? avg_stall_be*100.0/avg_cycles : 0.0) << "%)" << std::endl;
        std::cout << "Avg Branch misses: " << avg_br_miss << std::endl;
        std::cout << "Avg Context switches: " << avg_ctx_sw << std::endl;
        std::cout << "Avg CPU migrations: " << avg_cpu_mig << std::endl;
        std::cout << "Avg Page faults: " << avg_page_fault << std::endl;
        std::cout << "Avg Cache refs: " << avg_cache_ref << ", misses: " << avg_cache_miss
                  << " (" << (avg_cache_ref > 0 ? avg_cache_miss*100.0/avg_cache_ref : 0.0) << "%)" << std::endl;
        std::cout << "Avg L1D loads: " << avg_l1d_loads << ", misses: " << avg_l1d_miss
                  << " (" << (avg_l1d_loads > 0 ? avg_l1d_miss*100.0/avg_l1d_loads : 0.0) << "%)" << std::endl;
        std::cout << "Avg LLC loads: " << avg_llc_loads << ", misses: " << avg_llc_miss
                  << " (" << (avg_llc_loads > 0 ? avg_llc_miss*100.0/avg_llc_loads : 0.0) << "%)" << std::endl;
        std::cout << "========================================" << std::endl;

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
