#include <faiss/IndexHNSWLib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <inttypes.h>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/omp_utils.h>

#ifndef NO_MANUAL_VECTORIZATION
#define NO_MANUAL_VECTORIZATION
#endif
#include <hnswlib/hnswalg.h>
#include <hnswlib/space_ip.h>
#include <hnswlib/space_l2.h>

namespace faiss {

struct IndexHNSWLib::DistanceComputerImpl : DistanceComputer {
    const Impl* impl;
    MetricType metric_type;
    int d;
    const float* query = nullptr;

    DistanceComputerImpl(const Impl* impl_in, MetricType metric, int dim)
            : impl(impl_in), metric_type(metric), d(dim) {}

    void set_query(const float* x) override {
        query = x;
    }

    float operator()(idx_t i) override;

    float symmetric_dis(idx_t i, idx_t j) override;
};

struct IndexHNSWLib::Impl {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;

    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 16;
    size_t random_seed = 123;
    bool allow_replace_deleted = false;

    Impl() = default;

    void initialize(int d, MetricType metric) {
        if (space) {
            return;
        }
        switch (metric) {
            case METRIC_L2:
                space.reset(new hnswlib::L2Space(d));
                break;
            case METRIC_INNER_PRODUCT:
                space.reset(new hnswlib::InnerProductSpace(d));
                break;
            default:
                FAISS_THROW_MSG("IndexHNSWLib only supports L2 and inner product metrics");
        }
    }

    void ensure_graph(int d, MetricType metric, size_t capacity) {
        initialize(d, metric);
        if (!graph) {
            size_t initial_cap = std::max<size_t>(capacity, 1);
            graph.reset(new hnswlib::HierarchicalNSW<float>(
                    space.get(),
                    initial_cap,
                    M,
                    std::max<size_t>(ef_construction, M),
                    random_seed,
                    allow_replace_deleted));
            graph->setEf(std::max<size_t>(ef_search, 1));
            graph->ef_construction_ = std::max<size_t>(ef_construction, M);
        } else {
            ensure_capacity(capacity);
            graph->ef_construction_ = std::max<size_t>(ef_construction, M);
        }
    }

    void ensure_capacity(size_t need) {
        if (!graph) {
            return;
        }
        size_t current = graph->getMaxElements();
        if (need > current) {
            size_t new_cap = current;
            while (new_cap < need) {
                new_cap = std::max<size_t>(new_cap * 2, need);
            }
            graph->resizeIndex(new_cap);
        }
    }

    const float* get_vector(idx_t key) const {
        if (!graph) {
            return nullptr;
        }
        std::lock_guard<std::mutex> guard(graph->label_lookup_lock);
        auto it = graph->label_lookup_.find(static_cast<hnswlib::labeltype>(key));
        if (it == graph->label_lookup_.end()) {
            return nullptr;
        }
        hnswlib::tableint internal = it->second;
        if (graph->isMarkedDeleted(internal)) {
            return nullptr;
        }
        return reinterpret_cast<float*>(graph->getDataByInternalId(internal));
    }

    void clear() {
        if (graph) {
            graph->clear();
            graph.reset();
        }
        space.reset();
    }

    void apply_search_params(size_t k) const {
        if (graph) {
            size_t effective = std::max<size_t>(ef_search, k);
            graph->setEf(effective);
        }
    }

    size_t size() const {
        return graph ? graph->getCurrentElementCount() : 0;
    }
};

IndexHNSWLib::IndexHNSWLib(
        int d,
        int M,
        MetricType metric,
        size_t efConstruction,
        size_t efSearch,
        size_t random_seed)
        : Index(d, metric),
          impl_(std::make_unique<Impl>()) {
    impl_->M = std::max<size_t>(1, M);
    impl_->ef_construction = std::max<size_t>(impl_->M, efConstruction);
    impl_->ef_search = std::max<size_t>(1, efSearch);
    impl_->random_seed = random_seed;
    is_trained = true;
}

IndexHNSWLib::~IndexHNSWLib() = default;

void IndexHNSWLib::ensure_trained() const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexHNSWLib not trained");
}

void IndexHNSWLib::ensure_ready_for_add(idx_t n) {
    impl_->ensure_graph(d, metric_type, static_cast<size_t>(ntotal + n));
}

void IndexHNSWLib::train(idx_t, const float*) {
    is_trained = true;
}

void IndexHNSWLib::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(x || n == 0, "Null input in IndexHNSWLib::add");
    ensure_trained();
    if (n == 0) {
        return;
    }
    impl_->ensure_graph(d, metric_type, static_cast<size_t>(ntotal + n));
    impl_->ensure_capacity(static_cast<size_t>(ntotal + n));
    impl_->graph->ef_construction_ = std::max<size_t>(impl_->ef_construction, impl_->M);

    const idx_t base = ntotal;
    size_t num_threads = std::max<size_t>(1, faiss::omp_get_max_threads());

#pragma omp parallel for schedule(static) if (num_threads > 1) num_threads(num_threads)
    for (idx_t i = 0; i < n; ++i) {
        const float* vec = x + i * d;
        hnswlib::labeltype label = static_cast<hnswlib::labeltype>(base + i);
        impl_->graph->addPoint(vec, label);
    }
    ntotal += n;
}

void IndexHNSWLib::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters*) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(distances);
    FAISS_THROW_IF_NOT(labels);
    if (n == 0) {
        return;
    }
    if (!impl_->graph || impl_->size() == 0) {
        std::fill(labels, labels + n * k, -1);
        if (is_similarity_metric(metric_type)) {
            std::fill(distances, distances + n * k, -std::numeric_limits<float>::infinity());
        } else {
            std::fill(distances, distances + n * k, std::numeric_limits<float>::infinity());
        }
        return;
    }

    impl_->apply_search_params(static_cast<size_t>(k));

    for (idx_t qi = 0; qi < n; ++qi) {
        const float* query = x + qi * d;
        auto result = impl_->graph->searchKnn(query, static_cast<size_t>(k));

        size_t base = static_cast<size_t>(qi * k);
        size_t count = result.size();
        size_t idx = count;

        while (!result.empty()) {
            const auto& top = result.top();
            --idx;
            labels[base + idx] = static_cast<idx_t>(top.second);
            float dist = top.first;
            if (metric_type == METRIC_INNER_PRODUCT) {
                distances[base + idx] = 1.0f - dist;
            } else {
                distances[base + idx] = dist;
            }
            result.pop();
        }

        if (count < static_cast<size_t>(k)) {
            float pad_value = is_similarity_metric(metric_type)
                    ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity();
            for (size_t r = count; r < static_cast<size_t>(k); ++r) {
                labels[base + r] = -1;
                distances[base + r] = pad_value;
            }
        }
    }
}

void IndexHNSWLib::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("IndexHNSWLib::range_search is not implemented");
}

void IndexHNSWLib::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(recons, "Null buffer in IndexHNSWLib::reconstruct");
    const float* vec = impl_->get_vector(key);
    FAISS_THROW_IF_NOT_FMT(vec, "Key %" PRId64 " not found in IndexHNSWLib", key);
    std::memcpy(recons, vec, sizeof(float) * d);
}

void IndexHNSWLib::reset() {
    impl_->clear();
    ntotal = 0;
    is_trained = true;
}

DistanceComputer* IndexHNSWLib::get_distance_computer() const {
    ensure_trained();
    return new DistanceComputerImpl(impl_.get(), metric_type, d);
}

void IndexHNSWLib::set_efSearch(size_t ef) {
    impl_->ef_search = std::max<size_t>(1, ef);
    if (impl_->graph) {
        impl_->graph->setEf(impl_->ef_search);
    }
}

void IndexHNSWLib::set_efConstruction(size_t ef) {
    impl_->ef_construction = std::max<size_t>(impl_->M, ef);
    if (impl_->graph) {
        impl_->graph->ef_construction_ = impl_->ef_construction;
    }
}

void IndexHNSWLib::set_random_seed(size_t seed) {
    impl_->random_seed = seed;
    if (impl_->graph) {
        impl_->graph->level_generator_.seed(seed);
        impl_->graph->update_probability_generator_.seed(seed + 1);
    }
}

size_t IndexHNSWLib::efSearch() const {
    return impl_->ef_search;
}

size_t IndexHNSWLib::efConstruction() const {
    return impl_->ef_construction;
}

size_t IndexHNSWLib::random_seed() const {
    return impl_->random_seed;
}

bool IndexHNSWLib::save(const std::string& path) const {
    if (!impl_->graph) {
        return false;
    }
    impl_->graph->saveIndex(path);
    return true;
}

bool IndexHNSWLib::load(const std::string& path, size_t max_elements) {
    ensure_trained();
    impl_->initialize(d, metric_type);
    try {
        impl_->graph.reset(new hnswlib::HierarchicalNSW<float>(
                impl_->space.get(),
                path,
                false,
                max_elements,
                impl_->allow_replace_deleted));
    } catch (...) {
        impl_->graph.reset();
        ntotal = 0;
        return false;
    }
    ntotal = impl_->graph->cur_element_count.load();
    impl_->graph->setEf(std::max<size_t>(impl_->ef_search, 1));
    return true;
}

size_t IndexHNSWLib::allocated_bytes() const {
    if (!impl_->graph) {
        return 0;
    }
    const auto* graph = impl_->graph.get();
    const size_t count = graph->cur_element_count.load();
    size_t total = count * graph->size_data_per_element_;

    for (size_t i = 0; i < count; ++i) {
        int level = graph->element_levels_[i];
        if (level > 0) {
            total += static_cast<size_t>(level) * graph->size_links_per_element_;
        }
    }
    return total;
}

const hnswlib::HierarchicalNSW<float>* IndexHNSWLib::raw_graph() const {
    return impl_->graph.get();
}

hnswlib::HierarchicalNSW<float>* IndexHNSWLib::raw_graph() {
    return impl_->graph.get();
}

float IndexHNSWLib::DistanceComputerImpl::operator()(idx_t i) {
    FAISS_THROW_IF_NOT(query);
    const float* vec = impl->get_vector(i);
    if (!vec) {
        return std::numeric_limits<float>::infinity();
    }
    if (metric_type == METRIC_INNER_PRODUCT) {
        return -fvec_inner_product(query, vec, d);
    }
    return fvec_L2sqr(query, vec, d);
}

float IndexHNSWLib::DistanceComputerImpl::symmetric_dis(idx_t i, idx_t j) {
    const float* vi = impl->get_vector(i);
    const float* vj = impl->get_vector(j);
    if (!vi || !vj) {
        return std::numeric_limits<float>::infinity();
    }
    if (metric_type == METRIC_INNER_PRODUCT) {
        return -fvec_inner_product(vi, vj, d);
    }
    return fvec_L2sqr(vi, vj, d);
}

} // namespace faiss
