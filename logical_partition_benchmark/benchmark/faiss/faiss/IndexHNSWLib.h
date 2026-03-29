/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <faiss/Index.h>

namespace hnswlib {
template <typename T>
class HierarchicalNSW;
} // namespace hnswlib

namespace faiss {

/** HNSW index backed by the hnswlib implementation. This variant stores the
 *  vectors directly inside the HNSW structure to allow head-to-head comparisons
 *  with the Faiss implementation that keeps vectors in a separate storage.
 */
struct IndexHNSWLib : Index {
    IndexHNSWLib(
            int d,
            int M = 16,
            MetricType metric = METRIC_L2,
            size_t efConstruction = 200,
            size_t efSearch = 16,
            size_t random_seed = 123);

    ~IndexHNSWLib() override;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    DistanceComputer* get_distance_computer() const override;

    void set_efSearch(size_t ef);
    void set_efConstruction(size_t ef);
    void set_random_seed(size_t seed);

    size_t efSearch() const;
    size_t efConstruction() const;
    size_t random_seed() const;

    bool save(const std::string& path) const;
    bool load(const std::string& path, size_t max_elements = 0);

    size_t allocated_bytes() const;

    const hnswlib::HierarchicalNSW<float>* raw_graph() const;
    hnswlib::HierarchicalNSW<float>* raw_graph();

private:
protected:
    struct Impl;
    struct DistanceComputerImpl;

private:
    std::unique_ptr<Impl> impl_;

    void ensure_ready_for_add(idx_t n);
    void ensure_trained() const;
};

} // namespace faiss
