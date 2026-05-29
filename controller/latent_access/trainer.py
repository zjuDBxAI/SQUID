"""Latent access trainer.

This module learns a compact access basis from the document-tenant visibility
matrix and couples it with semantic structure through both group-wise and
neighbor-wise regularization.

The implementation keeps the public trainer interface stable while upgrading the
internals from a clustering proxy to an explicit sparse alternating factorization
of the form:

    A ~= ZB

where:
- A: document x tenant binary access matrix
- Z: document x atom sparse non-negative assignments
- B: atom x tenant non-negative tenant compatibility matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency fallback
    KMeans = None

from .repository import DocumentAccessRecord


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, min(max(float(q), 0.0), 1.0)))


@dataclass(slots=True)
class LatentAccessTrainingConfig:
    atom_count: int = 64
    semantic_cell_count: int = 64
    residual_quantile: float = 0.9
    access_weight: float = 1.0
    semantic_weight: float = 0.0
    semantic_knn: int = 8
    semantic_knn_weight: float = 0.2
    max_atoms_per_semantic_cell: int = 4
    min_partition_documents: int = 4
    max_tenant_dimensions: int = 2048
    sparsity: int = 8
    max_iterations: int = 25
    z_inner_iterations: int = 4
    z_step_scale: float = 1.0
    momentum_weight: float = 0.1
    ridge_lambda: float = 1e-3
    min_atom_support: float = 1.0
    revive_dead_atoms: bool = True
    revive_every: int = 3
    revive_residual_quantile: float = 0.85
    convergence_tol: float = 1e-4
    random_state: int = 42


@dataclass(slots=True)
class TrainedLatentAccessModel:
    config: LatentAccessTrainingConfig
    tenant_ids: tuple[int, ...]
    atom_tenant_weights: np.ndarray # B
    document_atom_assignments: np.ndarray 
    document_residuals: np.ndarray
    document_atom_weights: np.ndarray   # Z
    semantic_group_assignments: np.ndarray 
    training_metadata: dict[str, float | int | str] = field(default_factory=dict)

    def tenant_atom_scores(self, tenant_id: int) -> np.ndarray:
        try:
            tenant_index = self.tenant_ids.index(int(tenant_id))
        except ValueError:
            return np.zeros(self.atom_tenant_weights.shape[0], dtype=np.float32)
        return self.atom_tenant_weights[:, tenant_index].astype(np.float32, copy=False)

    def residual_threshold(self) -> float:
        return _quantile(self.document_residuals, self.config.residual_quantile)

    def reconstruct_access_matrix(self) -> np.ndarray:
        return self.document_atom_weights @ self.atom_tenant_weights


class PrototypeLatentAccessTrainer:
    """Train a sparse latent access basis with semantic graph regularization.

    The class name is kept for compatibility with existing imports, but the
    implementation is now a true alternating factorization trainer rather than a
    clustering-only proxy.
    """

    def __init__(self, config: Optional[LatentAccessTrainingConfig] = None) -> None:
        self.config = config or LatentAccessTrainingConfig()

    def fit(self, records: list[DocumentAccessRecord]) -> TrainedLatentAccessModel:
        if not records:
            raise ValueError("Cannot train latent access model without records")

        tenant_ids = tuple(sorted({tenant_id for record in records for tenant_id in record.tenant_ids}))
        tenant_to_index = {tenant_id: index for index, tenant_id in enumerate(tenant_ids)}
        if self.config.max_tenant_dimensions and len(tenant_ids) > int(self.config.max_tenant_dimensions):
            raise ValueError(
                f"Tenant count {len(tenant_ids)} exceeds configured limit "
                f"{self.config.max_tenant_dimensions}; increase max_tenant_dimensions first."
            )

        semantic_matrix = _normalize_rows(np.vstack([record.vector for record in records]).astype(np.float32))
        access_matrix = self._build_access_matrix(records, tenant_ids=tenant_ids, tenant_to_index=tenant_to_index)
        effective_atom_count = max(1, min(int(self.config.atom_count), len(records)))
        semantic_group_assignments = self._cluster_semantic_groups(
            semantic_matrix,
            requested_count=min(int(self.config.semantic_cell_count), len(records)),
        )
        neighbor_indices, neighbor_weights = self._build_semantic_neighbor_graph(semantic_matrix)
        z = self._initialize_document_weights(
            access_matrix=access_matrix,
            semantic_matrix=semantic_matrix,
            semantic_group_assignments=semantic_group_assignments,
            neighbor_indices=neighbor_indices,
            neighbor_weights=neighbor_weights,
            atom_count=effective_atom_count,
        )

        b = self._update_atom_tenant_weights(access_matrix=access_matrix, z=z)
        semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
        neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
        initial_objective = self._objective_value(
            access_matrix=access_matrix,
            z=z,
            b=b,
            semantic_group_assignments=semantic_group_assignments,
            semantic_targets=semantic_targets,
            neighbor_targets=neighbor_targets,
        )

        previous_objective = initial_objective
        iterations_run = 0
        revived_atoms_total = 0
        final_delta = 0.0
        for iteration in range(int(self.config.max_iterations)):
            iterations_run = iteration + 1
            b = self._update_atom_tenant_weights(access_matrix=access_matrix, z=z)
            semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
            neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
            candidate_z = self._update_document_weights(
                access_matrix=access_matrix,
                z=z,
                b=b,
                semantic_group_assignments=semantic_group_assignments,
                semantic_targets=semantic_targets,
                neighbor_targets=neighbor_targets,
            )
            revived_atoms = 0
            if self.config.revive_dead_atoms and (iteration + 1) % max(1, int(self.config.revive_every)) == 0:
                candidate_z, revived_atoms = self._revive_or_reseed_dead_atoms(
                    access_matrix=access_matrix,
                    z=candidate_z,
                    b=b,
                )
            candidate_semantic_targets = self._compute_semantic_group_targets(candidate_z, semantic_group_assignments)
            candidate_neighbor_targets = self._compute_neighbor_targets(candidate_z, neighbor_indices, neighbor_weights)
            objective = self._objective_value(
                access_matrix=access_matrix,
                z=candidate_z,
                b=b,
                semantic_group_assignments=semantic_group_assignments,
                semantic_targets=candidate_semantic_targets,
                neighbor_targets=candidate_neighbor_targets,
            )
            if objective > previous_objective:
                blended_z = self._project_sparse_rows(0.5 * (candidate_z + z))
                blended_semantic_targets = self._compute_semantic_group_targets(blended_z, semantic_group_assignments)
                blended_neighbor_targets = self._compute_neighbor_targets(blended_z, neighbor_indices, neighbor_weights)
                blended_objective = self._objective_value(
                    access_matrix=access_matrix,
                    z=blended_z,
                    b=b,
                    semantic_group_assignments=semantic_group_assignments,
                    semantic_targets=blended_semantic_targets,
                    neighbor_targets=blended_neighbor_targets,
                )
                if blended_objective <= previous_objective:
                    candidate_z = blended_z
                    objective = blended_objective
                    candidate_semantic_targets = blended_semantic_targets
                    candidate_neighbor_targets = blended_neighbor_targets
                else:
                    candidate_z = z.copy()
                    objective = previous_objective
                    revived_atoms = 0
                    candidate_semantic_targets = semantic_targets
                    candidate_neighbor_targets = neighbor_targets
            final_delta = float(np.mean(np.abs(candidate_z - z)))
            z = candidate_z
            revived_atoms_total += revived_atoms
            if (
                abs(previous_objective - objective) <= self.config.convergence_tol
                and final_delta <= self.config.convergence_tol
            ):
                previous_objective = objective
                break
            previous_objective = objective

        b = self._update_atom_tenant_weights(access_matrix=access_matrix, z=z)
        reconstruction = z @ b
        residuals = np.mean(np.abs(access_matrix - reconstruction), axis=1)
        assignments = np.argmax(z, axis=1)
        atom_supports = np.sum(z, axis=0)
        avg_nonzero_atoms = float(np.count_nonzero(z) / max(z.shape[0], 1))
        active_atom_count = int(np.count_nonzero(atom_supports >= float(self.config.min_atom_support)))
        semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
        neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
        final_objective = self._objective_value(
            access_matrix=access_matrix,
            z=z,
            b=b,
            semantic_group_assignments=semantic_group_assignments,
            semantic_targets=semantic_targets,
            neighbor_targets=neighbor_targets,
        )

        return TrainedLatentAccessModel(
            config=self.config,
            tenant_ids=tenant_ids,
            atom_tenant_weights=b.astype(np.float32, copy=False),
            document_atom_assignments=assignments.astype(np.int32, copy=False),
            document_residuals=residuals.astype(np.float32, copy=False),
            document_atom_weights=z.astype(np.float32, copy=False),
            semantic_group_assignments=semantic_group_assignments.astype(np.int32, copy=False),
            training_metadata={
                "document_count": len(records),
                "tenant_count": len(tenant_ids),
                "atom_count": effective_atom_count,
                "active_atom_count": active_atom_count,
                "iterations": iterations_run,
                "residual_threshold": _quantile(residuals, self.config.residual_quantile),
                "reconstruction_mae": float(np.mean(residuals)),
                "initial_objective": float(initial_objective),
                "final_objective": float(final_objective),
                "final_delta": float(final_delta),
                "avg_nonzero_atoms": avg_nonzero_atoms,
                "configured_min_atom_support": float(self.config.min_atom_support),
                "avg_atom_support": float(np.mean(atom_supports)) if atom_supports.size else 0.0,
                "observed_min_atom_support": float(np.min(atom_supports)) if atom_supports.size else 0.0,
                "observed_max_atom_support": float(np.max(atom_supports)) if atom_supports.size else 0.0,
                "dead_atom_revivals": int(revived_atoms_total),
                "semantic_group_count": int(semantic_targets.shape[0]),
                "semantic_knn": int(neighbor_indices.shape[1] if neighbor_indices.ndim == 2 else 0),
            },
        )

    def infer(self, records: list[DocumentAccessRecord], reference_model: TrainedLatentAccessModel) -> TrainedLatentAccessModel:
        if not records:
            raise ValueError("Cannot infer latent access model without records")

        tenant_ids = reference_model.tenant_ids
        tenant_to_index = {tenant_id: index for index, tenant_id in enumerate(tenant_ids)}
        semantic_matrix = _normalize_rows(np.vstack([record.vector for record in records]).astype(np.float32))
        access_matrix = self._build_access_matrix(
            records,
            tenant_ids=tenant_ids,
            tenant_to_index=tenant_to_index,
            ignore_missing_tenants=True,
        )
        semantic_group_assignments = self._cluster_semantic_groups(
            semantic_matrix,
            requested_count=min(int(self.config.semantic_cell_count), len(records)),
        )
        neighbor_indices, neighbor_weights = self._build_semantic_neighbor_graph(semantic_matrix)
        z = self._initialize_document_weights_from_atom_weights(
            access_matrix=access_matrix,
            b=reference_model.atom_tenant_weights,
            semantic_group_assignments=semantic_group_assignments,
            neighbor_indices=neighbor_indices,
            neighbor_weights=neighbor_weights,
        )

        b = reference_model.atom_tenant_weights.astype(np.float32, copy=True)
        semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
        neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
        initial_objective = self._objective_value(
            access_matrix=access_matrix,
            z=z,
            b=b,
            semantic_group_assignments=semantic_group_assignments,
            semantic_targets=semantic_targets,
            neighbor_targets=neighbor_targets,
        )

        previous_objective = initial_objective
        iterations_run = 0
        final_delta = 0.0
        for iteration in range(int(self.config.max_iterations)):
            iterations_run = iteration + 1
            semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
            neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
            candidate_z = self._update_document_weights(
                access_matrix=access_matrix,
                z=z,
                b=b,
                semantic_group_assignments=semantic_group_assignments,
                semantic_targets=semantic_targets,
                neighbor_targets=neighbor_targets,
            )
            candidate_semantic_targets = self._compute_semantic_group_targets(candidate_z, semantic_group_assignments)
            candidate_neighbor_targets = self._compute_neighbor_targets(candidate_z, neighbor_indices, neighbor_weights)
            objective = self._objective_value(
                access_matrix=access_matrix,
                z=candidate_z,
                b=b,
                semantic_group_assignments=semantic_group_assignments,
                semantic_targets=candidate_semantic_targets,
                neighbor_targets=candidate_neighbor_targets,
            )
            if objective > previous_objective:
                blended_z = self._project_sparse_rows(0.5 * (candidate_z + z))
                blended_semantic_targets = self._compute_semantic_group_targets(blended_z, semantic_group_assignments)
                blended_neighbor_targets = self._compute_neighbor_targets(blended_z, neighbor_indices, neighbor_weights)
                blended_objective = self._objective_value(
                    access_matrix=access_matrix,
                    z=blended_z,
                    b=b,
                    semantic_group_assignments=semantic_group_assignments,
                    semantic_targets=blended_semantic_targets,
                    neighbor_targets=blended_neighbor_targets,
                )
                if blended_objective <= previous_objective:
                    candidate_z = blended_z
                    objective = blended_objective
                else:
                    candidate_z = z.copy()
                    objective = previous_objective
            final_delta = float(np.mean(np.abs(candidate_z - z)))
            z = candidate_z
            if (
                abs(previous_objective - objective) <= self.config.convergence_tol
                and final_delta <= self.config.convergence_tol
            ):
                previous_objective = objective
                break
            previous_objective = objective

        reconstruction = z @ b
        residuals = np.mean(np.abs(access_matrix - reconstruction), axis=1)
        assignments = np.argmax(z, axis=1)
        unseen_tenants = sorted(
            {
                int(tenant_id)
                for record in records
                for tenant_id in record.tenant_ids
                if int(tenant_id) not in tenant_to_index
            }
        )
        return TrainedLatentAccessModel(
            config=self.config,
            tenant_ids=tenant_ids,
            atom_tenant_weights=b.astype(np.float32, copy=False),
            document_atom_assignments=assignments.astype(np.int32, copy=False),
            document_residuals=residuals.astype(np.float32, copy=False),
            document_atom_weights=z.astype(np.float32, copy=False),
            semantic_group_assignments=semantic_group_assignments.astype(np.int32, copy=False),
            training_metadata={
                "document_count": len(records),
                "tenant_count": len(tenant_ids),
                "atom_count": int(b.shape[0]),
                "iterations": iterations_run,
                "residual_threshold": _quantile(residuals, self.config.residual_quantile),
                "reconstruction_mae": float(np.mean(residuals)),
                "initial_objective": float(initial_objective),
                "final_objective": float(previous_objective),
                "final_delta": float(final_delta),
                "inference_mode": "fixed_atom_tenant_weights",
                "reference_document_count": int(reference_model.training_metadata.get("document_count", 0)),
                "ignored_unseen_tenant_count": len(unseen_tenants),
            },
        )

    def _build_access_matrix(
        self,
        records: list[DocumentAccessRecord],
        *,
        tenant_ids: tuple[int, ...],
        tenant_to_index: dict[int, int],
        ignore_missing_tenants: bool = False,
    ) -> np.ndarray:
        matrix = np.zeros((len(records), len(tenant_ids)), dtype=np.float32)
        for row_index, record in enumerate(records):
            for tenant_id in record.tenant_ids:
                normalized_tenant_id = int(tenant_id)
                if normalized_tenant_id not in tenant_to_index:
                    if ignore_missing_tenants:
                        continue
                    raise KeyError(f"Tenant {normalized_tenant_id} not found in tenant_to_index")
                matrix[row_index, tenant_to_index[normalized_tenant_id]] = 1.0
        return matrix

    def _build_hybrid_matrix(
        self,
        *,
        access_matrix: np.ndarray,
        semantic_matrix: np.ndarray,
    ) -> np.ndarray:
        normalized_access = _normalize_rows(access_matrix)
        weighted_access = normalized_access * float(self.config.access_weight)
        weighted_semantic = semantic_matrix * float(self.config.semantic_weight)
        return np.hstack([weighted_access, weighted_semantic]).astype(np.float32, copy=False)

    def _cluster_assignments(self, feature_matrix: np.ndarray, *, atom_count: int) -> np.ndarray:
        effective_atoms = max(1, min(int(atom_count), feature_matrix.shape[0]))
        if KMeans is None:
            return np.arange(feature_matrix.shape[0], dtype=np.int32) % effective_atoms
        estimator = KMeans(n_clusters=effective_atoms, random_state=self.config.random_state, n_init=10)
        return estimator.fit_predict(feature_matrix).astype(np.int32, copy=False)

    def _cluster_semantic_groups(self, semantic_matrix: np.ndarray, *, requested_count: int) -> np.ndarray:
        effective_groups = max(1, min(int(requested_count), semantic_matrix.shape[0]))
        if KMeans is None:
            return np.arange(semantic_matrix.shape[0], dtype=np.int32) % effective_groups
        estimator = KMeans(n_clusters=effective_groups, random_state=self.config.random_state, n_init=10)
        return estimator.fit_predict(semantic_matrix).astype(np.int32, copy=False)

    def _build_semantic_neighbor_graph(self, semantic_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        document_count = semantic_matrix.shape[0]
        neighbor_count = min(max(int(self.config.semantic_knn), 0), max(document_count - 1, 0))
        if document_count <= 1 or neighbor_count == 0:
            return (
                np.zeros((document_count, 0), dtype=np.int32),
                np.zeros((document_count, 0), dtype=np.float32),
            )

        neighbor_indices = np.zeros((document_count, neighbor_count), dtype=np.int32)
        neighbor_weights = np.zeros((document_count, neighbor_count), dtype=np.float32)
        chunk_size = max(64, min(2048, document_count))
        for start in range(0, document_count, chunk_size):
            stop = min(start + chunk_size, document_count)
            similarities = semantic_matrix[start:stop] @ semantic_matrix.T
            local_rows = np.arange(stop - start)
            similarities[local_rows, start + local_rows] = -np.inf
            local_neighbor_indices = np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
            local_neighbor_scores = np.take_along_axis(similarities, local_neighbor_indices, axis=1)
            order = np.argsort(local_neighbor_scores, axis=1)[:, ::-1]
            local_neighbor_indices = np.take_along_axis(local_neighbor_indices, order, axis=1)
            local_neighbor_scores = np.take_along_axis(local_neighbor_scores, order, axis=1)
            local_neighbor_scores = np.clip(local_neighbor_scores, 0.0, None)
            row_sums = np.sum(local_neighbor_scores, axis=1, keepdims=True)
            empty_rows = row_sums[:, 0] <= 1e-8
            if np.any(empty_rows):
                local_neighbor_scores[empty_rows] = 1.0
                row_sums = np.sum(local_neighbor_scores, axis=1, keepdims=True)
            neighbor_indices[start:stop] = local_neighbor_indices.astype(np.int32, copy=False)
            neighbor_weights[start:stop] = (local_neighbor_scores / row_sums).astype(np.float32, copy=False)
        return neighbor_indices, neighbor_weights

    def _initialize_document_weights(
        self,
        *,
        access_matrix: np.ndarray,
        semantic_matrix: np.ndarray,
        semantic_group_assignments: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        atom_count: int,
    ) -> np.ndarray:
        hybrid_matrix = self._build_hybrid_matrix(access_matrix=access_matrix, semantic_matrix=semantic_matrix)
        assignments = self._cluster_assignments(hybrid_matrix, atom_count=atom_count)
        z = np.zeros((access_matrix.shape[0], atom_count), dtype=np.float32)
        z[np.arange(access_matrix.shape[0]), assignments] = 1.0
        semantic_targets = self._compute_semantic_group_targets(z, semantic_group_assignments)
        neighbor_targets = self._compute_neighbor_targets(z, neighbor_indices, neighbor_weights)
        smoothed = z.copy()
        if semantic_targets.size:
            smoothed += float(self.config.semantic_weight) * semantic_targets[semantic_group_assignments]
        if neighbor_targets.size:
            smoothed += float(self.config.semantic_knn_weight) * neighbor_targets
        return self._project_sparse_rows(smoothed)

    def _initialize_document_weights_from_atom_weights(
        self,
        *,
        access_matrix: np.ndarray,
        b: np.ndarray,
        semantic_group_assignments: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
    ) -> np.ndarray:
        access_affinity = np.maximum(access_matrix @ b.T, 0.0)
        semantic_targets = self._compute_semantic_group_targets(access_affinity, semantic_group_assignments)
        neighbor_targets = self._compute_neighbor_targets(access_affinity, neighbor_indices, neighbor_weights)
        seeded = access_affinity.copy()
        if semantic_targets.size:
            seeded += float(self.config.semantic_weight) * semantic_targets[semantic_group_assignments]
        if neighbor_targets.size:
            seeded += float(self.config.semantic_knn_weight) * neighbor_targets
        return self._project_sparse_rows(seeded)

    def _compute_semantic_group_targets(self, z: np.ndarray, semantic_group_assignments: np.ndarray) -> np.ndarray:
        group_count = int(semantic_group_assignments.max()) + 1 if semantic_group_assignments.size else 0
        targets = np.zeros((group_count, z.shape[1]), dtype=np.float32)
        counts = np.zeros(group_count, dtype=np.int32)
        for row_index, group_id in enumerate(semantic_group_assignments):
            targets[int(group_id)] += z[row_index]
            counts[int(group_id)] += 1
        nonzero_mask = counts > 0
        if np.any(nonzero_mask):
            targets[nonzero_mask] /= counts[nonzero_mask][:, None]
        return np.clip(targets, 0.0, None).astype(np.float32, copy=False)

    def _compute_neighbor_targets(
        self,
        z: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
    ) -> np.ndarray:
        if neighbor_indices.size == 0 or neighbor_weights.size == 0:
            return np.zeros_like(z, dtype=np.float32)
        return np.sum(z[neighbor_indices] * neighbor_weights[:, :, None], axis=1).astype(np.float32, copy=False)

    def _update_atom_tenant_weights(self, *, access_matrix: np.ndarray, z: np.ndarray) -> np.ndarray:
        gram = z.T @ z
        gram += float(self.config.ridge_lambda) * np.eye(gram.shape[0], dtype=np.float32)
        rhs = z.T @ access_matrix
        solved = np.linalg.solve(gram, rhs)
        return np.clip(solved, 0.0, 1.0).astype(np.float32, copy=False)

    def _estimate_z_step(self, b: np.ndarray) -> float:
        if b.size == 0:
            return 1.0
        gram = b @ b.T
        spectral_radius = float(np.linalg.norm(gram, ord=2)) if gram.size else 0.0
        lipschitz = (
            spectral_radius
            + float(self.config.semantic_weight)
            + float(self.config.semantic_knn_weight)
            + float(self.config.momentum_weight)
            + 1e-6
        )
        return float(self.config.z_step_scale) / lipschitz

    def _update_document_weights(
        self,
        *,
        access_matrix: np.ndarray,
        z: np.ndarray,
        b: np.ndarray,
        semantic_group_assignments: np.ndarray,
        semantic_targets: np.ndarray,
        neighbor_targets: np.ndarray,
    ) -> np.ndarray:
        current = z.astype(np.float32, copy=True)
        step = self._estimate_z_step(b)
        group_targets = (
            semantic_targets[semantic_group_assignments]
            if semantic_targets.size
            else np.zeros_like(current, dtype=np.float32)
        )
        local_targets = (
            neighbor_targets.astype(np.float32, copy=False)
            if neighbor_targets.size
            else np.zeros_like(current, dtype=np.float32)
        )

        for _ in range(max(1, int(self.config.z_inner_iterations))):
            reconstruction_gradient = (current @ b - access_matrix) @ b.T
            gradient = reconstruction_gradient
            if semantic_targets.size:
                gradient = gradient + float(self.config.semantic_weight) * (current - group_targets)
            if neighbor_targets.size:
                gradient = gradient + float(self.config.semantic_knn_weight) * (current - local_targets)
            if self.config.momentum_weight > 0:
                gradient = gradient + float(self.config.momentum_weight) * (current - z)
            current = np.maximum(current - step * gradient, 0.0)
            current = self._project_sparse_rows(current)
        return current.astype(np.float32, copy=False)

    def _revive_or_reseed_dead_atoms(
        self,
        *,
        access_matrix: np.ndarray,
        z: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        atom_supports = np.sum(z, axis=0)
        dead_atoms = np.flatnonzero(atom_supports < float(self.config.min_atom_support))
        if dead_atoms.size == 0 or not self.config.revive_dead_atoms:
            return z.astype(np.float32, copy=False), 0

        reconstruction = z @ b
        residuals = np.mean(np.abs(access_matrix - reconstruction), axis=1)
        residual_floor = _quantile(residuals, float(self.config.revive_residual_quantile))
        candidate_order = np.argsort(residuals)[::-1]
        reseeded = z.astype(np.float32, copy=True)
        used_documents: set[int] = set()
        revived_atoms = 0
        for atom_id in dead_atoms:
            candidate = None
            for document_index in candidate_order:
                document_index = int(document_index)
                if document_index in used_documents:
                    continue
                if float(residuals[document_index]) < residual_floor:
                    break
                candidate = document_index
                break
            if candidate is None:
                break
            reseeded[candidate] = 0.0
            reseeded[candidate, int(atom_id)] = 1.0
            used_documents.add(candidate)
            revived_atoms += 1
        if revived_atoms == 0:
            return z.astype(np.float32, copy=False), 0
        return self._project_sparse_rows(reseeded), revived_atoms

    def _project_sparse_rows(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.astype(np.float32, copy=False)
        projected = np.zeros_like(matrix, dtype=np.float32)
        max_nonzero = max(1, min(int(self.config.sparsity), matrix.shape[1]))
        for row_index in range(matrix.shape[0]):
            row = np.asarray(matrix[row_index], dtype=np.float32)
            positive = np.flatnonzero(row > 0)
            if positive.size == 0:
                projected[row_index, int(np.argmax(row))] = 1.0
                continue
            if positive.size > max_nonzero:
                top_local = np.argpartition(row[positive], -max_nonzero)[-max_nonzero:]
                selected = positive[top_local]
            else:
                selected = positive
            weights = row[selected]
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                projected[row_index, int(selected[np.argmax(weights)])] = 1.0
                continue
            projected[row_index, selected] = weights / weight_sum
        return projected.astype(np.float32, copy=False)

    def _objective_value(
        self,
        *,
        access_matrix: np.ndarray,
        z: np.ndarray,
        b: np.ndarray,
        semantic_group_assignments: np.ndarray,
        semantic_targets: np.ndarray,
        neighbor_targets: np.ndarray,
    ) -> float:
        reconstruction = z @ b
        reconstruction_loss = float(np.mean((access_matrix - reconstruction) ** 2))
        if semantic_targets.size:
            semantic_loss = float(np.mean((z - semantic_targets[semantic_group_assignments]) ** 2))
        else:
            semantic_loss = 0.0
        if neighbor_targets.size:
            neighbor_loss = float(np.mean((z - neighbor_targets) ** 2))
        else:
            neighbor_loss = 0.0
        sparsity_loss = float(np.count_nonzero(z) / max(z.shape[0], 1))
        return (
            reconstruction_loss
            + float(self.config.semantic_weight) * semantic_loss
            + float(self.config.semantic_knn_weight) * neighbor_loss
            + 1e-3 * sparsity_loss
        )
