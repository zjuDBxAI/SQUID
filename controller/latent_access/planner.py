"""Planning helpers for semantic-cell x latent-atom partitioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency fallback
    KMeans = None

from .repository import DocumentAccessRecord
from .trainer import TrainedLatentAccessModel, _normalize_rows


@dataclass(slots=True)
class LatentAccessPartition:
    partition_id: str
    semantic_cell_id: int
    latent_atom_id: Optional[int]
    residual_flag: bool
    document_ids: tuple[int, ...]
    tenant_ids: tuple[int, ...]
    document_count: int
    vector_count: int
    metadata: dict[str, float | int | str | list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class LatentAccessPlan:
    partitions: list[LatentAccessPartition]
    semantic_centroids: dict[int, np.ndarray]
    semantic_assignments: np.ndarray
    model: TrainedLatentAccessModel
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class LatentAccessPlanner:
    """Build physical partitions from a trained latent access model."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)

    def build_plan(
        self,
        records: list[DocumentAccessRecord],
        model: TrainedLatentAccessModel,
        *,
        document_block_counts: Optional[dict[int, int]] = None,
    ) -> LatentAccessPlan:
        if not records:
            raise ValueError("Cannot build latent access plan without records")

        vectors = _normalize_rows(np.vstack([record.vector for record in records]).astype(np.float32))
        semantic_assignments, semantic_centroids = self._cluster_semantic_cells(
            vectors=vectors,
            requested_count=model.config.semantic_cell_count,
        )
        residual_threshold = model.residual_threshold()
        residual_mask = model.document_residuals > residual_threshold
        stable_atoms_by_cell = self._select_stable_atoms_by_cell(
            model=model,
            semantic_assignments=semantic_assignments,
            residual_mask=residual_mask,
        )

        grouped: dict[tuple[int, Optional[int], bool], list[int]] = {}
        redirected_to_stable_atom = 0
        redirected_to_residual = 0
        stable_atom_cells = sum(1 for atoms in stable_atoms_by_cell.values() if atoms)

        for index, record in enumerate(records):
            semantic_cell_id = int(semantic_assignments[index])
            residual_flag = bool(residual_mask[index])
            latent_atom_id: Optional[int] = None
            if not residual_flag:
                preferred_atom_id = int(model.document_atom_assignments[index])
                stable_atoms = stable_atoms_by_cell.get(semantic_cell_id, ())
                if preferred_atom_id in stable_atoms:
                    latent_atom_id = preferred_atom_id
                else:
                    reassigned_atom_id = self._select_supported_atom(
                        atom_weights=model.document_atom_weights[index],
                        candidate_atom_ids=stable_atoms,
                    )
                    if reassigned_atom_id is None:
                        residual_flag = True
                        redirected_to_residual += 1
                    else:
                        latent_atom_id = reassigned_atom_id
                        redirected_to_stable_atom += 1
            grouped.setdefault((semantic_cell_id, latent_atom_id, residual_flag), []).append(index)

        partitions: list[LatentAccessPartition] = []
        ordered_partition_keys = sorted(
            grouped.keys(),
            key=lambda item: (int(item[0]), bool(item[2]), -1 if item[1] is None else int(item[1])),
        )
        for semantic_cell_id, latent_atom_id, residual_flag in ordered_partition_keys:
            member_indices = grouped[(semantic_cell_id, latent_atom_id, residual_flag)]
            member_records = [records[index] for index in member_indices]
            document_ids = tuple(sorted(record.document_id for record in member_records))
            tenant_ids = tuple(sorted({tenant_id for record in member_records for tenant_id in record.tenant_ids}))
            if document_block_counts is None:
                vector_count = len(document_ids)
            else:
                vector_count = sum(int(document_block_counts.get(document_id, 1)) for document_id in document_ids)
            residual_values = [float(model.document_residuals[index]) for index in member_indices]
            centroid = semantic_centroids[semantic_cell_id]
            partition_id = (
                f"cell_{semantic_cell_id}_residual"
                if residual_flag
                else f"cell_{semantic_cell_id}_atom_{latent_atom_id}"
            )
            partitions.append(
                LatentAccessPartition(
                    partition_id=partition_id,
                    semantic_cell_id=semantic_cell_id,
                    latent_atom_id=latent_atom_id,
                    residual_flag=residual_flag,
                    document_ids=document_ids,
                    tenant_ids=tenant_ids,
                    document_count=len(document_ids),
                    vector_count=int(vector_count),
                    metadata={
                        "semantic_centroid": centroid.astype(float).tolist(),
                        "average_residual": float(np.mean(residual_values) if residual_values else 0.0),
                        "max_residual": float(np.max(residual_values) if residual_values else 0.0),
                        "route_prior": float(len(document_ids) / max(len(records), 1)),
                    },
                )
            )

        singleton_partition_count = sum(1 for partition in partitions if partition.document_count == 1)
        residual_partition_count = sum(1 for partition in partitions if partition.residual_flag)
        return LatentAccessPlan(
            partitions=partitions,
            semantic_centroids=semantic_centroids,
            semantic_assignments=semantic_assignments,
            model=model,
            metadata={
                "document_count": len(records),
                "partition_count": len(partitions),
                "semantic_cell_count": len(semantic_centroids),
                "residual_threshold": residual_threshold,
                "residual_partition_count": residual_partition_count,
                "singleton_partition_count": singleton_partition_count,
                "stable_atom_cells": stable_atom_cells,
                "redirected_to_stable_atom": redirected_to_stable_atom,
                "redirected_to_residual": redirected_to_residual,
            },
        )

    def _select_stable_atoms_by_cell(
        self,
        *,
        model: TrainedLatentAccessModel,
        semantic_assignments: np.ndarray,
        residual_mask: np.ndarray,
    ) -> dict[int, tuple[int, ...]]:
        min_partition_documents = max(1, int(model.config.min_partition_documents))
        max_atoms_per_cell = max(1, int(model.config.max_atoms_per_semantic_cell))
        dominant_doc_counts: dict[int, dict[int, int]] = {}
        dominant_weight_sums: dict[int, dict[int, float]] = {}

        for index in range(len(semantic_assignments)):
            if bool(residual_mask[index]):
                continue
            cell_id = int(semantic_assignments[index])
            atom_id = int(model.document_atom_assignments[index])
            dominant_doc_counts.setdefault(cell_id, {})
            dominant_weight_sums.setdefault(cell_id, {})
            dominant_doc_counts[cell_id][atom_id] = dominant_doc_counts[cell_id].get(atom_id, 0) + 1
            dominant_weight_sums[cell_id][atom_id] = dominant_weight_sums[cell_id].get(atom_id, 0.0) + float(
                model.document_atom_weights[index, atom_id]
            )

        stable_atoms_by_cell: dict[int, tuple[int, ...]] = {}
        for cell_id, atom_counts in dominant_doc_counts.items():
            ranked_atoms = sorted(
                atom_counts.keys(),
                key=lambda atom_id: (
                    int(atom_counts[atom_id]),
                    float(dominant_weight_sums[cell_id].get(atom_id, 0.0)),
                    -int(atom_id),
                ),
                reverse=True,
            )
            stable_atoms = [
                int(atom_id)
                for atom_id in ranked_atoms
                if int(atom_counts[atom_id]) >= min_partition_documents
            ][:max_atoms_per_cell]
            stable_atoms_by_cell[cell_id] = tuple(stable_atoms)
        return stable_atoms_by_cell

    def _select_supported_atom(
        self,
        *,
        atom_weights: np.ndarray,
        candidate_atom_ids: tuple[int, ...],
    ) -> Optional[int]:
        if not candidate_atom_ids:
            return None
        candidate_array = np.asarray(candidate_atom_ids, dtype=np.int32)
        candidate_weights = atom_weights[candidate_array]
        if candidate_weights.size == 0:
            return None
        best_local_index = int(np.argmax(candidate_weights))
        best_weight = float(candidate_weights[best_local_index])
        if best_weight <= 0.0:
            return None
        return int(candidate_array[best_local_index])

    def _cluster_semantic_cells(
        self,
        *,
        vectors: np.ndarray,
        requested_count: int,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        cell_count = max(1, min(int(requested_count), vectors.shape[0]))
        if KMeans is None:
            assignments = np.arange(vectors.shape[0], dtype=np.int32) % cell_count
            centroids = {
                cell_id: vectors[assignments == cell_id].mean(axis=0)
                for cell_id in range(cell_count)
            }
            return assignments, centroids

        estimator = KMeans(n_clusters=cell_count, random_state=self.random_state, n_init=10)
        assignments = estimator.fit_predict(vectors)
        centroids = {cell_id: estimator.cluster_centers_[cell_id].astype(np.float32) for cell_id in range(cell_count)}
        return assignments.astype(np.int32, copy=False), centroids
