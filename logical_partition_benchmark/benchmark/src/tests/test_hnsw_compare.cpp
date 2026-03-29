#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexHNSWLib.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>

namespace {

std::vector<float> generate_dataset(int d, size_t n) {
    std::vector<float> data(n * static_cast<size_t>(d));
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            data[i * static_cast<size_t>(d) + j] =
                    static_cast<float>(i * 0.5 + j * 0.1);
        }
    }
    return data;
}

bool almost_equal(float a, float b, float tol = 1e-4f) {
    if (std::isinf(a) && std::isinf(b)) {
        return (std::signbit(a) == std::signbit(b));
    }
    return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

void compare_indices(
        faiss::Index& faiss_index,
        faiss::Index& lib_index,
        const std::vector<float>& vectors,
        const std::vector<float>& queries,
        int d,
        int k) {
    const size_t nq = queries.size() / static_cast<size_t>(d);
    std::vector<faiss::idx_t> labels_faiss(nq * k);
    std::vector<faiss::idx_t> labels_lib(nq * k);
    std::vector<float> distances_faiss(nq * k);
    std::vector<float> distances_lib(nq * k);

    faiss_index.search(nq, queries.data(), k, distances_faiss.data(), labels_faiss.data());
    lib_index.search(nq, queries.data(), k, distances_lib.data(), labels_lib.data());

    for (size_t qi = 0; qi < nq; ++qi) {
        for (int ri = 0; ri < k; ++ri) {
            size_t idx = qi * k + ri;
            if (labels_faiss[idx] != labels_lib[idx]) {
                std::cerr << "Label mismatch at query " << qi << " rank " << ri
                          << ": faiss=" << labels_faiss[idx]
                          << " hnswlib=" << labels_lib[idx] << std::endl;
                throw std::runtime_error("Label mismatch");
            }
            if (!almost_equal(distances_faiss[idx], distances_lib[idx])) {
                std::cerr << "Distance mismatch at query " << qi << " rank " << ri
                          << ": faiss=" << distances_faiss[idx]
                          << " hnswlib=" << distances_lib[idx] << std::endl;
                throw std::runtime_error("Distance mismatch");
            }
        }
    }
}

} // namespace

int main() {
    try {
        const int d = 8;
        const size_t nb = 64;
        const size_t nq = 10;
        const int M = 16;
        const int efConstruction = 60;
        const int efSearch = 32;
        const int k = 5;

        auto base = generate_dataset(d, nb);
        auto queries = generate_dataset(d, nq);

        // Faiss (separate storage).
        faiss::IndexHNSWFlat faiss_index_l2(d, M, faiss::METRIC_L2);
        faiss_index_l2.hnsw.efConstruction = efConstruction;
        faiss_index_l2.hnsw.efSearch = efSearch;
        faiss_index_l2.add(nb, base.data());

        // hnswlib-backed (integrated storage).
        faiss::IndexHNSWLib lib_index_l2(d, M, faiss::METRIC_L2, efConstruction, efSearch, 123);
        lib_index_l2.add(nb, base.data());

        compare_indices(faiss_index_l2, lib_index_l2, base, queries, d, k);

        // Repeat the same check for inner product.
        faiss::IndexHNSWFlat faiss_index_ip(d, M, faiss::METRIC_INNER_PRODUCT);
        faiss_index_ip.hnsw.efConstruction = efConstruction;
        faiss_index_ip.hnsw.efSearch = efSearch;
        faiss_index_ip.add(nb, base.data());

        faiss::IndexHNSWLib lib_index_ip(
                d,
                M,
                faiss::METRIC_INNER_PRODUCT,
                efConstruction,
                efSearch,
                321);
        lib_index_ip.add(nb, base.data());

        compare_indices(faiss_index_ip, lib_index_ip, base, queries, d, k);

        // Sanity check: distance computer consistency.
        std::unique_ptr<faiss::DistanceComputer> dc(lib_index_l2.get_distance_computer());
        dc->set_query(queries.data());
        auto dist = (*dc)(0);
        auto gt_dist = faiss::fvec_L2sqr(queries.data(), base.data(), d);
        if (!almost_equal(dist, gt_dist)) {
            std::cerr << "DistanceComputer mismatch: " << dist << " != " << gt_dist << std::endl;
            throw std::runtime_error("DistanceComputer validation failed");
        }

        std::cout << "HNSW comparison test passed." << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << std::endl;
        return 1;
    }
}
