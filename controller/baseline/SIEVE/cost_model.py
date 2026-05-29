from __future__ import annotations

import math

from .common import round_to_multiple_of_four


class SieveCostModel:
    def __init__(
        self,
        *,
        dataset_size: int,
        m: int = 16,
        bitvector_cutoff: int = 1000,
        ef_search: int = 10,
        k: int = 10,
        heterogeneous_indexing: bool = True,
        heterogeneous_search: bool = True,
        query_correlation_constant: float = 0.5,
        ef_search_scaling_constant: float = 3.0,
    ) -> None:
        self.dataset_size = max(1, int(dataset_size))
        self.m = max(1, int(m))
        self.bitvector_cutoff = max(2, int(bitvector_cutoff))
        self.ef_search = max(1, int(ef_search))
        self.k = max(1, int(k))
        self.heterogeneous_indexing = bool(heterogeneous_indexing)
        self.heterogeneous_search = bool(heterogeneous_search)
        self.query_correlation_constant = float(query_correlation_constant)
        self.ef_search_scaling_constant = max(1e-9, float(ef_search_scaling_constant))

    def downscaled_m(self, cardinality: int) -> int:
        cardinality = max(1, int(cardinality))
        if not self.heterogeneous_indexing:
            return int(self.m)
        if self.dataset_size <= 1 or cardinality <= 1:
            return 4
        ratio = math.log10(cardinality) / math.log10(self.dataset_size)
        return round_to_multiple_of_four(ratio * self.m)

    def downscaled_ef_search(self, cardinality: int, *, ef_search: int | None = None, k: int | None = None) -> int:
        effective_ef = self.ef_search if ef_search is None else max(1, int(ef_search))
        effective_k = self.k if k is None else max(1, int(k))
        cardinality = max(1, int(cardinality))
        if not self.heterogeneous_search:
            return max(effective_k, effective_ef)
        if self.dataset_size <= 1 or cardinality <= 1:
            return effective_k
        ratio = math.log10(cardinality) / math.log10(self.dataset_size)
        return max(effective_k, int(ratio * effective_ef))

    def bf_search_cost(self, query_cardinality: int) -> float:
        query_cardinality = max(1, int(query_cardinality))
        return float(query_cardinality) * math.log(float(self.bitvector_cutoff)) / float(self.bitvector_cutoff)

    def upward_search_cost(self, parent_cardinality: int, query_cardinality: int) -> float:
        parent_cardinality = max(1, int(parent_cardinality))
        query_cardinality = max(1, int(query_cardinality))
        if self.heterogeneous_search:
            new_ef = self.downscaled_ef_search(parent_cardinality)
        else:
            new_ef = self.ef_search
        new_ef_ratio = (self.k + ((float(new_ef) - self.k) / self.ef_search_scaling_constant)) / self.k
        return math.log(float(parent_cardinality)) * math.pow(
            float(parent_cardinality) / float(query_cardinality),
            self.query_correlation_constant,
        ) * new_ef_ratio

    def root_search_cost(self, query_cardinality: int) -> float:
        query_cardinality = max(1, int(query_cardinality))
        new_ef_ratio = (self.k + ((float(self.ef_search) - self.k) / self.ef_search_scaling_constant)) / self.k
        return math.log(float(self.dataset_size)) * math.pow(
            float(self.dataset_size) / float(query_cardinality),
            self.query_correlation_constant,
        ) * new_ef_ratio

    def scaled_partition_size(self, cardinality: int) -> int:
        cardinality = max(1, int(cardinality))
        effective_m = self.downscaled_m(cardinality) if self.heterogeneous_indexing else self.m
        return int(cardinality * (effective_m + 50) / 82)

    def budget_units(self, index_vector_budget: float) -> int:
        root_scaled_size = self.scaled_partition_size(self.dataset_size)
        if float(index_vector_budget) <= 0:
            return 0
        raw_budget = float(index_vector_budget) * float(root_scaled_size)
        return max(0, int(raw_budget))

