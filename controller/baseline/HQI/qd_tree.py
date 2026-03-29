from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from typing import Literal
from itertools import count

import numpy as np
import psycopg2
from psycopg2 import extras, sql
from sklearn.cluster import KMeans


project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.append(project_root)

from services.config import get_db_connection, get_maintenance_settings  # pylint: disable=wrong-import-position
from controller.baseline.pg_row_security.row_level_security import (  # pylint: disable=wrong-import-position
    get_db_connection_for_many_users,
)


logger = logging.getLogger(__name__)

_env_log_level = os.environ.get("QD_TREE_LOG_LEVEL") or os.environ.get("PYTHONLOGLEVEL")
if _env_log_level:
    level = getattr(logging, _env_log_level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setLevel(level)
        _handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(_handler)
        logger.propagate = False


PredicateType = Literal["role", "centroid"]

def _load_qd_tree_config() -> Dict[str, str]:
    config_path = Path(project_root) / "qd_tree_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as cfg_file:
            data = json.load(cfg_file)
        if isinstance(data, dict):
            return data
        logger.warning("qd_tree_config.json should contain a JSON object; ignoring contents.")
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load qd_tree_config.json: %s", exc)
        return {}


_QD_TREE_CONFIG = _load_qd_tree_config()

_DEFAULT_PICKLE_PATH = _QD_TREE_CONFIG.get("pickle_path") or os.environ.get(
    "QD_TREE_PICKLE_PATH"
)
if _DEFAULT_PICKLE_PATH is None:
    _DEFAULT_PICKLE_PATH = str(Path.home() / "qd-tree-artifacts" / "qd_tree.pkl")

DEFAULT_QD_TREE_PATH = Path(_DEFAULT_PICKLE_PATH).expanduser().resolve()
DEFAULT_QD_TREE_PARTITION_PREFIX = _QD_TREE_CONFIG.get(
    "partition_prefix",
    os.environ.get("QD_TREE_PARTITION_PREFIX", "documentblocks_qdtree_partition"),
)


def get_qd_tree_config() -> Dict[str, str]:
    return _QD_TREE_CONFIG.copy()


_QD_TREE_CACHE: Optional["QDTreeNode"] = None
_QD_TREE_CACHE_PATH: Optional[Path] = None
_QD_TREE_ROLE_INDEX: Dict[str, Set[str]] = {}
_QD_TREE_LEAF_MAP: Dict[str, QDTreeNode] = {}

@dataclass
class Document:
    """
    Represents a single document block entry in the QD-tree.
    """

    block_id: int
    doc_id: int
    vector: np.ndarray
    accessible_roles: Set[str]
    centroid_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.block_id = int(self.block_id)
        self.doc_id = int(self.doc_id)
        raw_vector = self.vector
        if raw_vector is None:
            raise ValueError(f"Vector is None for document_id={self.doc_id}, block_id={self.block_id}")

        if isinstance(raw_vector, memoryview):
            vector_array = np.frombuffer(raw_vector, dtype=np.float32)
        elif isinstance(raw_vector, (bytes, bytearray)):
            vector_array = np.frombuffer(raw_vector, dtype=np.float32)
        elif isinstance(raw_vector, str):
            try:
                vector_array = np.asarray(json.loads(raw_vector), dtype=np.float32)
            except json.JSONDecodeError:
                vector_array = np.asarray(
                    [float(x) for x in raw_vector.strip("[]").split(",") if x],
                    dtype=np.float32,
                )
        elif hasattr(raw_vector, "tolist"):
            vector_array = np.asarray(raw_vector.tolist(), dtype=np.float32)
        else:
            try:
                vector_array = np.asarray(raw_vector, dtype=np.float32)
            except TypeError as exc:
                raise TypeError(
                    f"Unsupported vector type {type(raw_vector).__name__} for "
                    f"document_id={self.doc_id}, block_id={self.block_id}"
                ) from exc

        if vector_array.ndim != 1:
            vector_array = vector_array.ravel()
        if vector_array.size == 0:
            raise ValueError(
                f"Empty vector for document_id={self.doc_id}, block_id={self.block_id}"
            )

        self.vector = vector_array.astype(np.float32, copy=False)
        self.accessible_roles = {str(role) for role in self.accessible_roles}


@dataclass
class Query:
    """
    Represents a historical query used for workload-aware partitioning.
    """

    query_id: int
    user_id: int
    vector: np.ndarray
    accessible_doc_ids: Set[int]
    top_k: int
    metadata: Dict[str, Union[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)
        self.accessible_doc_ids = {int(doc_id) for doc_id in self.accessible_doc_ids}
@dataclass(frozen=True)
class Predicate:
    predicate_type: PredicateType  # Either "role" or "centroid".
    value: Union[str, int]

    def evaluate(self, document: Document) -> bool:
        if self.predicate_type == "role":
            return str(self.value) in document.accessible_roles
        if self.predicate_type == "centroid":
            return document.centroid_id == int(self.value)
        raise ValueError(f"Unknown predicate type: {self.predicate_type}")

    def __str__(self) -> str:
        return f"{self.predicate_type}={self.value}"


@dataclass
class QDTreeNode:
    depth: int
    documents: List[Document] = field(default_factory=list)
    split_predicate: Optional[Predicate] = None
    left_child: Optional["QDTreeNode"] = None
    right_child: Optional["QDTreeNode"] = None
    centroids: Optional[np.ndarray] = None  # Only populated at the root.
    partition_id: Optional[int] = None
    table_name: Optional[str] = None
    document_map: Dict[Tuple[int, int], Document] = field(default_factory=dict)
    document_lookup: Dict[Tuple[int, int], Document] = field(default_factory=dict)
    queries: List["Query"] = field(default_factory=list)
    required_roles: Set[str] = field(default_factory=set)
    document_roles: Set[str] = field(default_factory=set)

    def is_leaf(self) -> bool:
        return self.split_predicate is None

    def assign_document_map(self) -> None:
        self.document_map = {(doc.doc_id, doc.block_id): doc for doc in self.documents}
        if not hasattr(self, "document_roles") or self.document_roles is None:
            self.document_roles = set()
        if not self.document_roles:
            doc_roles: Set[str] = set()
            for doc in self.documents:
                doc_roles.update(doc.accessible_roles)
            self.document_roles = doc_roles


def run_kmeans(
    documents: Sequence[Document], num_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    if not documents:
        raise ValueError("Cannot run k-means without documents.")

    num_clusters = max(1, min(num_clusters, len(documents)))
    vectors = np.stack([doc.vector for doc in documents])
    logger.info(
        "Running KMeans: documents=%s, requested_clusters=%s, effective_clusters=%s",
        len(documents),
        num_clusters,
        num_clusters,
    )
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(vectors)
    for doc, label in zip(documents, labels):
        doc.centroid_id = int(label)
    return kmeans.cluster_centers_, kmeans


def generate_predicates(
    documents: Sequence[Document],
    num_centroids: int,
    include_centroid_predicates: bool = True,
) -> List[Predicate]:
    role_values = {role for doc in documents for role in doc.accessible_roles}
    predicates = [Predicate("role", role) for role in sorted(role_values)]
    if include_centroid_predicates:
        predicates.extend(Predicate("centroid", cid) for cid in range(num_centroids))
    logger.info(
        "Generated predicates: roles=%s, centroids=%s, total=%s",
        len(role_values),
        num_centroids if include_centroid_predicates else 0,
        len(predicates),
    )
    return predicates


def create_role_workload_from_documents(
    documents: Sequence[Document],
    top_k: int = 5,
) -> List[Query]:
    """
    Synthesize a workload where each role observed in the documents contributes a
    single query that references all documents available to that role. The query
    vectors are zeroed placeholders because role-only workloads focus on access
    control rather than vector similarity during tree construction.
    """
    if not documents:
        return []

    role_to_docs: Dict[str, Set[int]] = {}
    vector_dim = documents[0].vector.shape[0]
    zero_vector = np.zeros(vector_dim, dtype=np.float32)

    for doc in documents:
        for role in doc.accessible_roles:
            role_to_docs.setdefault(role, set()).add(doc.doc_id)

    synthesized_queries: List[Query] = []
    for idx, (role, doc_ids) in enumerate(sorted(role_to_docs.items())):
        if not doc_ids:
            continue
        try:
            user_id = int(role)
        except ValueError:
            # Use a negative surrogate ID to avoid collisions with genuine user IDs.
            user_id = -(idx + 1)
        synthesized_queries.append(
            Query(
                query_id=idx,
                user_id=user_id,
                vector=zero_vector.copy(),
                accessible_doc_ids=set(doc_ids),
                top_k=top_k,
                metadata={"role": role, "synthetic": True},
            )
        )

    return synthesized_queries


def export_qd_tree_to_dot(
    root: QDTreeNode,
    output_path: Union[str, Path],
    max_depth: Optional[int] = None,
    include_document_roles: bool = False,
) -> Path:
    """
    Serialize the QD-tree structure to a Graphviz DOT file for visualization.

    :param root: Root node of the QD-tree.
    :param output_path: Destination path for the DOT file.
    :param max_depth: Optional depth limit. Nodes deeper than this level are omitted.
    :param include_document_roles: If True, include the union of document roles for leaves.
    :return: Path to the written DOT file.
    """
    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    node_id_map: Dict[int, str] = {}
    node_labels: Dict[str, str] = {}
    edges: List[Tuple[str, str]] = []
    id_counter = count()

    stack: List[QDTreeNode] = [root]
    visited: Set[int] = set()

    def _format_roles(node: QDTreeNode) -> str:
        roles: Set[str] = set()
        if include_document_roles and getattr(node, "document_roles", None):
            roles.update(str(role) for role in node.document_roles)
        if getattr(node, "required_roles", None):
            roles.update(str(role) for role in node.required_roles)
        if not roles:
            return ""
        sorted_roles = sorted(roles)
        max_preview = 12
        preview = sorted_roles[:max_preview]
        suffix = ",…" if len(sorted_roles) > max_preview else ""
        return f"roles={','.join(preview)}{suffix}"

    while stack:
        node = stack.pop()
        node_key = id(node)
        if node_key in visited:
            continue
        if max_depth is not None and node.depth > max_depth:
            continue
        visited.add(node_key)

        node_identifier = node_id_map.setdefault(node_key, f"n{next(id_counter)}")
        label_parts = [f"depth {node.depth}"]

        if node.is_leaf():
            label_parts.append("leaf")
            if node.partition_id is not None:
                label_parts.append(f"pid={node.partition_id}")
            label_parts.append(f"size={len(node.documents)}")
            roles_str = _format_roles(node)
            if roles_str:
                label_parts.append(roles_str)
        else:
            predicate = node.split_predicate
            if predicate is not None:
                label_parts.append(f"{predicate.predicate_type}={predicate.value}")
            roles_str = _format_roles(node)
            if roles_str:
                label_parts.append(roles_str)

        node_labels[node_identifier] = "\\n".join(label_parts)

        for child in (node.left_child, node.right_child):
            if child is None:
                continue
            if max_depth is not None and child.depth > max_depth:
                continue
            child_key = id(child)
            child_identifier = node_id_map.setdefault(child_key, f"n{next(id_counter)}")
            edges.append((node_identifier, child_identifier))
            stack.append(child)

    with dest.open("w", encoding="utf-8") as fout:
        fout.write("digraph QDTree {\n")
        fout.write("  graph [rankdir=TB];\n")
        fout.write("  node [shape=box, style=\"rounded,filled\", fillcolor=\"#F5F5F5\", fontname=\"Helvetica\"];\n")
        fout.write("  edge [color=\"#888888\"];\n\n")
        for node_identifier, label in node_labels.items():
            fout.write(f"  {node_identifier} [label=\"{label}\"];\n")
        fout.write("\n")
        for parent, child in edges:
            fout.write(f"  {parent} -> {child};\n")
        fout.write("}\n")

    return dest


def partition_documents(
    documents: Sequence[Document], predicate: Predicate
) -> Tuple[List[Document], List[Document]]:
    left_docs, right_docs = [], []
    for doc in documents:
        (left_docs if predicate.evaluate(doc) else right_docs).append(doc)
    return left_docs, right_docs


def evaluate_split_cost(
    left_docs: Sequence[Document],
    right_docs: Sequence[Document],
    queries: Sequence[Query],
) -> Tuple[int, List[Query], List[Query]]:
    doc_left_ids = {doc.doc_id for doc in left_docs}
    doc_right_ids = {doc.doc_id for doc in right_docs}

    left_queries: List[Query] = []
    right_queries: List[Query] = []
    overlap = 0

    for query in queries:
        accessible = query.accessible_doc_ids
        left_needed = bool(doc_left_ids.intersection(accessible))
        right_needed = bool(doc_right_ids.intersection(accessible))
        if left_needed:
            left_queries.append(query)
        if right_needed:
            right_queries.append(query)
        if left_needed and right_needed:
            overlap += 1

    return overlap, left_queries, right_queries


def find_best_split(
    documents: Sequence[Document],
    predicates: Iterable[Predicate],
    queries: Sequence[Query],
    min_partition_size: int,
) -> Tuple[
    Optional[Predicate],
    Optional[List[Document]],
    Optional[List[Document]],
    Optional[List[Query]],
    Optional[List[Query]],
]:
    if not queries:
        best_score = -1.0
        best_predicate = None
        best_left_docs: Optional[List[Document]] = None
        best_right_docs: Optional[List[Document]] = None

        for predicate in predicates:
            left_docs, right_docs = partition_documents(documents, predicate)
            if len(left_docs) < min_partition_size or len(right_docs) < min_partition_size:
                continue
            split_ratio = len(left_docs) / len(documents)
            score = 1.0 - abs(0.5 - split_ratio)
            if score > best_score:
                best_score = score
                best_predicate = predicate
                best_left_docs, best_right_docs = left_docs, right_docs

        if best_predicate is None:
            logger.debug(
                "No valid split found at size=%s (min_partition_size=%s)",
                len(documents),
                min_partition_size,
            )
        else:
            logger.debug(
                "Selected predicate '%s' with score=%.4f (left=%s, right=%s) without workload",
                best_predicate,
                best_score,
                len(best_left_docs or []),
                len(best_right_docs or []),
            )
        return best_predicate, best_left_docs, best_right_docs, None, None

    best_score = -1.0
    best_predicate = None
    best_left: Optional[List[Document]] = None
    best_right: Optional[List[Document]] = None
    best_left_queries: Optional[List[Query]] = None
    best_right_queries: Optional[List[Query]] = None
    best_cost: Optional[int] = None

    for predicate in predicates:
        left_docs, right_docs = partition_documents(documents, predicate)
        if len(left_docs) < min_partition_size or len(right_docs) < min_partition_size:
            continue
        cost, left_queries, right_queries = evaluate_split_cost(left_docs, right_docs, queries)
        split_ratio = len(left_docs) / len(documents)
        score = 1.0 - abs(0.5 - split_ratio)
        if (
            best_cost is None
            or cost < best_cost
            or (cost == best_cost and score > best_score)
        ):
            best_cost = cost
            best_score = score
            best_predicate = predicate
            best_left, best_right = left_docs, right_docs
            best_left_queries, best_right_queries = left_queries, right_queries

    if best_predicate is None:
        logger.debug(
            "No valid split found at size=%s (min_partition_size=%s)",
            len(documents),
            min_partition_size,
        )
    else:
        logger.debug(
            "Selected predicate '%s' with cost=%s (left=%s, right=%s)",
            best_predicate,
            best_cost,
            len(best_left or []),
            len(best_right or []),
        )
    return best_predicate, best_left, best_right, best_left_queries, best_right_queries


def build_qd_tree(
    documents: Sequence[Document],
    queries: Sequence[Query],
    all_predicates: Sequence[Predicate],
    max_depth: Optional[int],
    min_partition_size: int,
    depth: int = 0,
    required_roles: Optional[Set[str]] = None,
) -> QDTreeNode:
    role_requirements = set(required_roles) if required_roles is not None else set()
    has_reached_depth_limit = max_depth is not None and depth >= max_depth
    if has_reached_depth_limit or len(documents) <= min_partition_size or not all_predicates:
        node = QDTreeNode(
            depth=depth,
            documents=list(documents),
            queries=list(queries),
            required_roles=set(role_requirements),
        )
        node.assign_document_map()
        logger.debug(
            "Created leaf node at depth=%s with %s blocks (max_depth=%s, min_partition_size=%s)",
            depth,
            len(documents),
            max_depth if max_depth is not None else "unbounded",
            min_partition_size,
        )
        return node

    predicate, left_docs, right_docs, left_queries, right_queries = find_best_split(
        documents, all_predicates, queries, min_partition_size
    )
    if predicate is None or left_docs is None or right_docs is None:
        node = QDTreeNode(
            depth=depth,
            documents=list(documents),
            queries=list(queries),
            required_roles=set(role_requirements),
        )
        node.assign_document_map()
        logger.debug(
            "Fallback leaf node at depth=%s with %s blocks due to invalid split",
            depth,
            len(documents),
        )
        return node

    remaining_preds = [p for p in all_predicates if p != predicate]
    node = QDTreeNode(
        depth=depth,
        split_predicate=predicate,
        queries=list(queries),
        required_roles=set(role_requirements),
    )
    left_queries = left_queries if left_queries is not None else list(queries)
    right_queries = right_queries if right_queries is not None else list(queries)

    left_required = set(role_requirements)
    right_required = set(role_requirements)
    if predicate.predicate_type == "role":
        left_required.add(str(predicate.value))

    node.left_child = build_qd_tree(
        left_docs,
        left_queries,
        remaining_preds,
        max_depth,
        min_partition_size,
        depth + 1,
        left_required,
    )
    node.right_child = build_qd_tree(
        right_docs,
        right_queries,
        remaining_preds,
        max_depth,
        min_partition_size,
        depth + 1,
        right_required,
    )
    return node


def extract_leaf_partitions(root: QDTreeNode) -> List[QDTreeNode]:
    leaves: List[QDTreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            leaves.append(node)
        else:
            if node.left_child is not None:
                stack.append(node.left_child)
            if node.right_child is not None:
                stack.append(node.right_child)
    logger.info("Extracted %s leaf partitions from QD-tree", len(leaves))
    return leaves


def build_rbac_qd_tree(
    documents: Sequence[Document],
    queries: Optional[Sequence[Query]],
    num_centroids: int,
    min_partition_size: int,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    include_centroid_predicates: bool = True,
) -> QDTreeNode:
    logger.info(
        "Building RBAC QD-tree (docs=%s, queries=%s, max_depth=%s, min_partition_size=%s)",
        len(documents),
        len(queries) if queries is not None else 0,
        max_depth if max_depth is not None else "unbounded",
        min_partition_size,
    )
    centroids, _ = run_kmeans(documents, num_centroids, random_state=random_state)
    predicates = generate_predicates(
        documents,
        len(centroids),
        include_centroid_predicates=include_centroid_predicates,
    )
    query_list = list(queries) if queries is not None else []
    root = build_qd_tree(documents, query_list, predicates, max_depth, min_partition_size)
    root.centroids = centroids
    leaves = extract_leaf_partitions(root)
    for leaf in leaves:
        leaf.assign_document_map()
    root.document_lookup = {(doc.doc_id, doc.block_id): doc for doc in documents}
    logger.info("Finished building RBAC QD-tree")
    return root


def get_nearest_centroid_id(centroids: np.ndarray, query_vector: np.ndarray) -> int:
    distances = np.linalg.norm(centroids - query_vector, axis=1)
    return int(distances.argmin())


def find_relevant_partitions(
    node: QDTreeNode,
    user_roles: Set[str],
    query_centroid_id: int,
) -> List[QDTreeNode]:
    if node.is_leaf():
        return [node]

    predicate = node.split_predicate
    if predicate is None:
        return [node]

    results: List[QDTreeNode] = []
    if predicate.predicate_type == "centroid":
        target_child = (
            node.left_child if query_centroid_id == int(predicate.value) else node.right_child
        )
        if target_child is not None:
            results.extend(find_relevant_partitions(target_child, user_roles, query_centroid_id))
    elif predicate.predicate_type == "role":
        role_value = str(predicate.value)
        if role_value in user_roles and node.left_child is not None:
            results.extend(find_relevant_partitions(node.left_child, user_roles, query_centroid_id))
        if node.right_child is not None:
            results.extend(find_relevant_partitions(node.right_child, user_roles, query_centroid_id))
    else:
        raise ValueError(f"Unsupported predicate type {predicate.predicate_type}")
    return results


def gather_role_accessible_partitions(node: QDTreeNode, user_roles: Set[str]) -> List[QDTreeNode]:
    """
    Collect every leaf partition that contains at least one document accessible to the user.
    """
    if not user_roles:
        return []

    if not _QD_TREE_ROLE_INDEX:
        leaves = extract_leaf_partitions(node)
        _rebuild_role_index(leaves, DEFAULT_QD_TREE_PARTITION_PREFIX)

    table_names: Set[str] = set()
    for role in user_roles:
        table_names.update(_QD_TREE_ROLE_INDEX.get(role, set()))

    return [
        _QD_TREE_LEAF_MAP[table_name]
        for table_name in table_names
        if table_name in _QD_TREE_LEAF_MAP
    ]


def partition_has_accessible_documents(partition: QDTreeNode, user_roles: Set[str]) -> bool:
    """
    Return True if at least one document inside this partition is visible to the user.
    """
    if not user_roles:
        return False
    return bool(partition.document_roles.intersection(user_roles))


def execute_hq_query(
    user_id: str,
    query_vector: Sequence[float],
    rbac_policy: Mapping[str, Iterable[str]],
    qd_tree_root: QDTreeNode,
    top_k: int = 5,
) -> List[Tuple[int, int]]:
    if qd_tree_root.centroids is None:
        raise ValueError("QD-tree root must contain centroid information.")
    user_roles = {str(role) for role in rbac_policy.get(user_id, [])}
    if not user_roles:
        logger.debug("No roles found for user_id=%s", user_id)
        return []

    query_vec = np.asarray(query_vector, dtype=np.float32)
    centroid_id = get_nearest_centroid_id(qd_tree_root.centroids, query_vec)
    target_partitions = find_relevant_partitions(qd_tree_root, user_roles, centroid_id)

    candidates: List[Tuple[int, float, Document]] = []
    for partition in target_partitions:
        if not partition.documents:
            continue
        vectors = np.stack([doc.vector for doc in partition.documents])
        distances = np.linalg.norm(vectors - query_vec, axis=1)
        for dist, doc in zip(distances, partition.documents):
            candidates.append((doc.doc_id, float(dist), doc))

    candidates.sort(key=lambda item: item[1])
    filtered_results: List[Tuple[int, int]] = []
    seen_blocks: Set[Tuple[int, int]] = set()
    for doc_id, _, doc in candidates:
        key = (doc_id, doc.block_id)
        if key in seen_blocks:
            continue
        if doc.accessible_roles.intersection(user_roles):
            filtered_results.append(key)
            seen_blocks.add(key)
        if len(filtered_results) == top_k:
            break
    logger.debug(
        "In-memory query finished for user_id=%s -> %s results",
        user_id,
        len(filtered_results),
    )
    return filtered_results


# ---------------------------------------------------------------------------
# Database integration helpers
# ---------------------------------------------------------------------------


def fetch_document_roles(conn) -> Dict[int, Set[str]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT document_id, array_agg(DISTINCT role_id)
        FROM PermissionAssignment
        GROUP BY document_id;
        """
    )
    roles_map: Dict[int, Set[str]] = {}
    for document_id, role_ids in cur.fetchall():
        if role_ids is None:
            roles_map[int(document_id)] = set()
        else:
            roles_map[int(document_id)] = {str(role_id) for role_id in role_ids}
    cur.close()
    logger.info("Fetched role assignments for %s documents", len(roles_map))
    return roles_map


def load_documents_from_db(
    batch_size: int = 5000,
    limit: Optional[int] = None,
) -> List[Document]:
    logger.info("Loading document blocks from DB (batch_size=%s, limit=%s)", batch_size, limit)
    conn = get_db_connection()
    documents: List[Document] = []
    try:
        roles_map = fetch_document_roles(conn)
        with conn.cursor(name="qd_tree_document_vectors") as cur:
            cur.itersize = batch_size
            cur.execute(
                """
                SELECT block_id, document_id, vector
                FROM documentblocks
                ORDER BY document_id, block_id;
                """
            )

            for idx, (block_id, document_id, vector) in enumerate(cur, start=1):
                doc_id = int(document_id)
                roles = roles_map.get(doc_id, set())
                documents.append(
                    Document(
                        block_id=int(block_id),
                        doc_id=doc_id,
                        vector=vector,
                        accessible_roles=roles,
                    )
                )
                if limit is not None and len(documents) >= limit:
                    logger.info("Reached limit=%s while streaming blocks", limit)
                    break
                if idx % (batch_size * 5) == 0:
                    logger.debug("Streamed %s blocks so far...", idx)
    finally:
        conn.close()
    logger.info("Loaded %s blocks from database", len(documents))
    return documents


def fetch_user_accessible_documents(user_ids: Sequence[int]) -> Dict[int, Set[int]]:
    user_ids = [int(uid) for uid in sorted(set(user_ids))]
    if not user_ids:
        return {}

    conn = get_db_connection()
    doc_map: Dict[int, Set[int]] = {uid: set() for uid in user_ids}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ur.user_id, pa.document_id
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON ur.role_id = pa.role_id
                WHERE ur.user_id = ANY(%s);
                """,
                (user_ids,),
            )
            for user_id, document_id in cur.fetchall():
                doc_map[int(user_id)].add(int(document_id))
    finally:
        conn.close()
    logger.info("Fetched accessible documents for %s users", len(doc_map))
    return doc_map


def _configure_partition_rls(cur, table_name: str) -> None:
    """
    Enable RLS on a QD-tree partition table and create the policy used at query time.
    """
    cur.execute("GRANT SELECT ON PermissionAssignment TO PUBLIC;")
    cur.execute("GRANT SELECT ON UserRoles TO PUBLIC;")
    cur.execute("GRANT SELECT ON DocumentBlocks TO PUBLIC;")
    cur.execute(sql.SQL("GRANT SELECT ON {} TO PUBLIC;").format(sql.Identifier(table_name)))
    cur.execute(
        sql.SQL("ALTER TABLE {} DISABLE ROW LEVEL SECURITY;").format(sql.Identifier(table_name))
    )
    cur.execute(
        sql.SQL("DROP POLICY IF EXISTS partition_access_policy ON {};").format(
            sql.Identifier(table_name)
        )
    )
    cur.execute(
        sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY;").format(sql.Identifier(table_name))
    )
    cur.execute(
        sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY;").format(sql.Identifier(table_name))
    )
    cur.execute(
        sql.SQL(
            """
            CREATE POLICY partition_access_policy ON {}
            FOR SELECT
            USING (
                EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON pa.role_id = ur.role_id
                    WHERE pa.document_id = {}.document_id
                      AND ur.user_id = current_user::int
                )
            );
            """
        ).format(sql.Identifier(table_name), sql.Identifier(table_name))
    )


def load_query_workload(
    json_path: str,
    limit: Optional[int] = None,
) -> List[Query]:
    with open(json_path, "r", encoding="utf-8") as fin:
        raw_queries = json.load(fin)

    if limit is not None:
        raw_queries = raw_queries[:limit]

    user_ids = [int(entry["user_id"]) for entry in raw_queries]
    accessible_docs_map = fetch_user_accessible_documents(user_ids)

    queries: List[Query] = []
    for idx, entry in enumerate(raw_queries):
        user_id = int(entry["user_id"])
        vector_raw = entry["query_vector"]
        if isinstance(vector_raw, str):
            vector = json.loads(vector_raw)
        else:
            vector = vector_raw
        accessible_docs = accessible_docs_map.get(user_id, set())
        metadata: Dict[str, Union[int, float]] = {}
        if "query_block_selectivity" in entry:
            metadata["query_block_selectivity"] = float(entry["query_block_selectivity"])
        top_k = int(entry.get("topk", entry.get("top_k", 5)))
        queries.append(
            Query(
                query_id=idx,
                user_id=user_id,
                vector=vector,
                accessible_doc_ids=accessible_docs,
                top_k=top_k,
                metadata=metadata,
            )
        )

    logger.info("Loaded %s queries from %s", len(queries), json_path)
    return queries


def build_qd_tree_from_database(
    queries: Optional[Sequence[Query]],
    num_centroids: int,
    min_partition_size: int,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    batch_size: int = 5000,
    limit: Optional[int] = None,
    include_centroid_predicates: bool = True,
) -> QDTreeNode:
    documents = load_documents_from_db(batch_size=batch_size, limit=limit)
    return build_rbac_qd_tree(
        documents,
        queries,
        num_centroids=num_centroids,
        min_partition_size=min_partition_size,
        max_depth=max_depth,
        random_state=random_state,
        include_centroid_predicates=include_centroid_predicates,
    )


def assign_partition_ids(
    leaves: Sequence[QDTreeNode],
    partition_prefix: str,
    start_partition_id: int = 0,
) -> None:
    for offset, leaf in enumerate(leaves, start=start_partition_id):
        pid = offset
        leaf.partition_id = pid
        leaf.table_name = f"{partition_prefix}_{pid}"
        logger.debug(
            "Assigned partition_id=%s to leaf with %s blocks", pid, len(leaf.documents)
        )


def _propagate_required_roles(node: QDTreeNode, inherited_roles: Set[str]) -> None:
    node.required_roles = set(inherited_roles)

    predicate = node.split_predicate
    if predicate is None:
        return

    left_roles = set(inherited_roles)
    right_roles = set(inherited_roles)

    if predicate.predicate_type == "role":
        left_roles.add(str(predicate.value))

    if node.left_child is not None:
        _propagate_required_roles(node.left_child, left_roles)
    if node.right_child is not None:
        _propagate_required_roles(node.right_child, right_roles)


def _rebuild_role_index(leaves: Sequence[QDTreeNode], prefix: str) -> None:
    global _QD_TREE_ROLE_INDEX, _QD_TREE_LEAF_MAP
    role_index: Dict[str, Set[str]] = {}
    leaf_map: Dict[str, QDTreeNode] = {}
    for leaf in leaves:
        table_name = leaf.table_name or f"{prefix}_{leaf.partition_id}"
        if not hasattr(leaf, "document_roles") or leaf.document_roles is None:
            leaf.document_roles = set()
        if not leaf.document_roles:
            for doc in leaf.documents:
                leaf.document_roles.update(doc.accessible_roles)
        leaf_map[table_name] = leaf
        for role in leaf.document_roles:
            role_index.setdefault(role, set()).add(table_name)
    _QD_TREE_ROLE_INDEX = role_index
    _QD_TREE_LEAF_MAP = leaf_map


def get_qd_tree_root(
    tree_path: Optional[str] = None,
    partition_prefix: Optional[str] = None,
) -> QDTreeNode:
    global _QD_TREE_CACHE, _QD_TREE_CACHE_PATH
    resolved_path = Path(tree_path).expanduser().resolve() if tree_path else DEFAULT_QD_TREE_PATH
    if _QD_TREE_CACHE is None or _QD_TREE_CACHE_PATH != resolved_path:
        logger.info("Loading QD-tree from '%s'", resolved_path)
        _QD_TREE_CACHE = load_qd_tree(str(resolved_path))
        _QD_TREE_CACHE_PATH = resolved_path
        _QD_TREE_ROLE_INDEX.clear()
        _QD_TREE_LEAF_MAP.clear()

    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    leaves = extract_leaf_partitions(_QD_TREE_CACHE)
    if any(leaf.table_name is None or not leaf.table_name.startswith(prefix) for leaf in leaves):
        assign_partition_ids(leaves, prefix)
    for leaf in leaves:
        if not leaf.document_map:
            leaf.assign_document_map()
        if not hasattr(leaf, "document_roles") or leaf.document_roles is None:
            leaf.document_roles = set()
        if not leaf.document_roles:
            for doc in leaf.documents:
                leaf.document_roles.update(doc.accessible_roles)
        if not hasattr(leaf, "required_roles") or leaf.required_roles is None:
            leaf.required_roles = set()
    _propagate_required_roles(_QD_TREE_CACHE, set())
    _rebuild_role_index(leaves, prefix)
    _QD_TREE_CACHE.partition_map = {leaf.partition_id: leaf for leaf in leaves}
    return _QD_TREE_CACHE


def get_role_partition_index() -> Dict[str, Set[str]]:
    """
    Return a copy of the cached role -> partition table mapping.
    Assumes `get_qd_tree_root` has been called to populate the cache.
    """
    return {role: set(tables) for role, tables in _QD_TREE_ROLE_INDEX.items()}


def _drop_existing_partitions(cur, prefix: str) -> None:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name LIKE %s;
        """,
        (f"{prefix}_%",),
    )
    existing = [row[0] for row in cur.fetchall()]
    for table_name in existing:
        logger.info("Dropping existing partition table '%s'", table_name)
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))


def _prepare_query_vector(query_vector: Sequence[float]) -> Tuple[np.ndarray, List[float]]:
    if isinstance(query_vector, str):
        arr = np.asarray(json.loads(query_vector), dtype=np.float32)
        return arr, arr.astype(np.float32).tolist()
    arr = np.asarray(query_vector, dtype=np.float32)
    return arr, arr.astype(np.float32).tolist()


def _fetch_user_roles(user_id: str) -> Set[str]:
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    try:
        cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s;", [user_id])
        roles = {str(row[0]) for row in cur.fetchall()}
    finally:
        cur.close()
        conn.close()
    return roles


def _collect_partition_document_ids_for_user(
    partition: QDTreeNode,
    user_roles: Set[str],
) -> Set[int]:
    """
    Gather the document IDs inside this partition that are visible to the user.
    """
    if not user_roles:
        return set()

    visible_docs: Set[int] = set()
    for doc in getattr(partition, "documents", []) or []:
        doc_roles = getattr(doc, "accessible_roles", set())
        if doc_roles and doc_roles.intersection(user_roles):
            visible_docs.add(int(doc.doc_id))
    return visible_docs


def _collect_relevant_partitions(
    root: QDTreeNode,
    user_roles: Set[str],
    query_vector: np.ndarray,
) -> List[QDTreeNode]:
    if root.centroids is None:
        raise ValueError("QD-tree root must contain centroid information.")
    centroid_id = get_nearest_centroid_id(root.centroids, query_vector)
    return find_relevant_partitions(root, user_roles, centroid_id)


def _merge_qd_tree_results(
    rows: Sequence[Tuple[int, int, Any, float]],
    top_k: int,
) -> List[Tuple[int, int, Any, float]]:
    filtered_results: List[Tuple[int, int, Any, float]] = []
    seen: Set[Tuple[int, int]] = set()
    sorted_results = sorted(rows, key=lambda item: float(item[3]))

    for block_id, document_id, block_content, distance in sorted_results:
        doc_id = int(document_id)
        blk_id = int(block_id)
        key = (doc_id, blk_id)
        if key in seen:
            continue
        filtered_results.append((blk_id, doc_id, block_content, float(distance)))
        seen.add(key)
        if len(filtered_results) == top_k:
            break
    return filtered_results


def _create_partition_indexes(cur, table_name: str, index_type: str = "hnsw") -> None:
    cur.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {} ON {} (document_id);
            """
        ).format(
            sql.Identifier(f"{table_name}_document_id_idx"),
            sql.Identifier(table_name),
        )
    )

    index_type_lower = index_type.lower()
    if index_type_lower == "hnsw":
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING hnsw (vector vector_l2_ops)
                WITH (m = 16, ef_construction = 64);
                """
            ).format(
                sql.Identifier(f"{table_name}_vector_idx"),
                sql.Identifier(table_name),
            )
        )
    elif index_type_lower == "ivfflat":
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING ivfflat (vector vector_l2_ops);
                """
            ).format(
                sql.Identifier(f"{table_name}_vector_idx"),
                sql.Identifier(table_name),
            )
        )


def _persist_partition_worker(args: Tuple[str, List[Tuple[int, int]], str]) -> Tuple[str, int]:
    table_name, pairs, index_type = args
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    block_id INT NOT NULL,
                    document_id INT NOT NULL REFERENCES Documents(document_id),
                    block_content BYTEA NOT NULL,
                    vector VECTOR(300),
                    PRIMARY KEY (block_id, document_id)
                );
                """
            ).format(sql.Identifier(table_name))
        )

        cur.execute("CREATE TEMP TABLE qd_tree_pairs(document_id INT, block_id INT) ON COMMIT DROP;")
        extras.execute_values(
            cur,
            "INSERT INTO qd_tree_pairs (document_id, block_id) VALUES %s",
            pairs,
            template="(%s, %s)",
        )
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {table} (block_id, document_id, block_content, vector)
                SELECT db.block_id, db.document_id, db.block_content, db.vector
                FROM documentblocks AS db
                JOIN qd_tree_pairs qp
                  ON db.document_id = qp.document_id AND db.block_id = qp.block_id;
                """
            ).format(table=sql.Identifier(table_name))
        )
        _create_partition_indexes(cur, table_name, index_type=index_type)
        _configure_partition_rls(cur, table_name)
        conn.commit()
        return table_name, len(pairs)
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to persist partition {table_name}: {exc}") from exc
    finally:
        cur.close()
        conn.close()


def persist_partitions_to_postgres(
    root: QDTreeNode,
    partition_prefix: str = "documentblocks_qdtree_partition",
    drop_existing: bool = True,
    index_type: str = "hnsw",
    workers: int = 1,
    apply_rls: bool = True,
) -> None:
    leaves = extract_leaf_partitions(root)
    assign_partition_ids(leaves, partition_prefix)

    if drop_existing:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            _drop_existing_partitions(cur, partition_prefix)
            conn.commit()
        finally:
            cur.close()
            conn.close()

    tasks: List[Tuple[str, List[Tuple[int, int]], str]] = []
    total_blocks = 0
    for leaf in leaves:
        if not leaf.documents:
            continue
        if leaf.partition_id is None:
            continue
        table_name = leaf.table_name or f"{partition_prefix}_{leaf.partition_id}"
        pairs = [(doc.doc_id, doc.block_id) for doc in leaf.documents]
        tasks.append((table_name, pairs, index_type))
        total_blocks += len(pairs)

    logger.info(
        "Persisting %s partitions (%s blocks) with %s worker(s)",
        len(tasks),
        total_blocks,
        workers,
    )

    if not tasks:
        logger.info("No partitions to persist.")
        return

    if workers <= 1:
        for table_name, pairs, index_type_value in tasks:
            logger.info("Persisting partition '%s' with %s blocks", table_name, len(pairs))
            _persist_partition_worker((table_name, pairs, index_type_value))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_persist_partition_worker, (table_name, pairs, index_type)): (
                    table_name,
                    len(pairs),
                )
                for table_name, pairs, index_type in tasks
            }
            for future in as_completed(futures):
                table_name, block_count = futures[future]
                future.result()
            logger.info("Persisted partition '%s' with %s blocks", table_name, block_count)

    logger.info("Finished persisting %s partitions to PostgreSQL", len(tasks))

    if apply_rls:
        logger.info("Re-applying RLS policies to partitions with prefix '%s'", partition_prefix)
        apply_rls_to_qdtree_partitions(partition_prefix)


def disable_rls_for_qdtree_partitions(partition_prefix: Optional[str] = None) -> None:
    """
    Disable RLS and drop the policy on each QD-tree partition table.
    """
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    tables = _list_qdtree_partition_tables(prefix)
    if not tables:
        logger.info("No QD-tree partition tables found with prefix '%s'", prefix)
        return

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for table_name in tables:
            cur.execute(
                sql.SQL("ALTER TABLE {} DISABLE ROW LEVEL SECURITY;").format(
                    sql.Identifier(table_name)
                )
            )
            cur.execute(
                sql.SQL("DROP POLICY IF EXISTS partition_access_policy ON {};").format(
                    sql.Identifier(table_name)
                )
            )
        conn.commit()
        logger.info("Disabled RLS for %s partition table(s) with prefix '%s'", len(tables), prefix)
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to disable RLS for prefix '{prefix}': {exc}") from exc
    finally:
        cur.close()
        conn.close()


def apply_rls_to_qdtree_partitions(partition_prefix: Optional[str] = None) -> None:
    """
    Retroactively enable RLS on existing QD-tree partitions.
    """
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    tables = _list_qdtree_partition_tables(prefix)
    if not tables:
        logger.info("No QD-tree partition tables found with prefix '%s'", prefix)
        return

    disable_rls_for_qdtree_partitions(prefix)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for table_name in tables:
            _configure_partition_rls(cur, table_name)
        conn.commit()
        logger.info("Configured RLS for %s partition table(s) with prefix '%s'", len(tables), prefix)
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to configure RLS for prefix '{prefix}': {exc}") from exc
    finally:
        cur.close()
        conn.close()


def qd_tree_search_statistics_sql(
    user_id: str,
    query_vector: Sequence[float],
    top_k: int,
    tree_root: QDTreeNode,
    partition_prefix: str,
) -> Tuple[List[Tuple[int, int, Any, float]], float]:
    user_roles = _fetch_user_roles(user_id)
    if not user_roles:
        logger.debug("No roles found for user_id=%s", user_id)
        return [], 0.0

    query_vec, query_param = _prepare_query_vector(query_vector)
    centroid_partitions = _collect_relevant_partitions(tree_root, user_roles, query_vec)
    centroid_with_access = [
        partition
        for partition in centroid_partitions
        if partition_has_accessible_documents(partition, user_roles)
    ]
    role_partitions = gather_role_accessible_partitions(tree_root, user_roles)
    partitions_dict: Dict[str, QDTreeNode] = {}
    for partition in centroid_with_access:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict[key] = partition
    for partition in role_partitions:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict.setdefault(key, partition)
    selected_partitions = list(partitions_dict.values())
    if centroid_with_access:
        logger.debug(
            "Centroid partitions with accessible docs: %s",
            [p.table_name or p.partition_id for p in centroid_with_access],
        )
    else:
        logger.debug("Centroid partitions had no accessible docs.")
    logger.debug(
        "Role-accessible partitions candidate: %s",
        [p.table_name or p.partition_id for p in role_partitions],
    )
    logger.debug(
        "Final partition list (order preserved): %s",
        [p.table_name or p.partition_id for p in selected_partitions],
    )
    partitions = selected_partitions

    if not partitions:
        logger.debug("No partitions matched for user_id=%s", user_id)
        return [], 0.0

    conn = get_db_connection()
    cur = conn.cursor()
    total_query_time = 0.0
    all_results: List[Tuple[int, int, Any, float]] = []

    try:
        import efconfig
        ef_search = getattr(efconfig, "ef_search", 40)
        cur.execute("SET max_parallel_workers_per_gather = 0;")
        cur.execute("SET jit = off;")
        cur.execute("SET enable_seqscan = off;")
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        logger.debug(
            "qd-tree SQL search configured: ef_search=%s, partitions=%s",
            ef_search,
            [p.table_name or p.partition_id for p in partitions],
        )

        for partition in partitions:
            table_name = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
            if table_name is None:
                continue
            logger.debug("Executing SQL search on partition %s", table_name)
            doc_id_filter = sorted(_collect_partition_document_ids_for_user(partition, user_roles))
            if not doc_id_filter:
                logger.debug(
                    "Skipping partition %s; no documents remain after role filtering",
                    table_name,
                )
                continue

            explain_query = sql.SQL(
                """
                EXPLAIN ANALYZE
                SELECT block_id, document_id, block_content,
                       vector <-> %s::vector AS distance
                FROM {}
                WHERE document_id = ANY(%s)
                ORDER BY distance
                LIMIT %s;
                """
            ).format(sql.Identifier(table_name))
            cur.execute(explain_query, [query_param, doc_id_filter, top_k])
            plan_rows = cur.fetchall()
            for row in plan_rows:
                line = row[0]
                if "Execution Time" in line:
                    parts = line.strip().split()
                    try:
                        total_query_time += float(parts[-2]) / 1000.0
                    except (ValueError, IndexError):
                        continue

            cur.execute(
                sql.SQL(
                    """
                    SELECT block_id, document_id, block_content,
                           vector <-> %s::vector AS distance
                    FROM {}
                    WHERE document_id = ANY(%s)
                    ORDER BY distance
                    LIMIT %s;
                    """
                ).format(sql.Identifier(table_name)),
                [query_param, doc_id_filter, top_k],
            )
            partition_rows = cur.fetchall()
            all_results.extend(partition_rows)
    finally:
        cur.close()
        conn.close()

    filtered_results = _merge_qd_tree_results(all_results, top_k)
    return filtered_results, total_query_time


def qd_tree_search_statistics_system(
    user_id: str,
    query_vector: Sequence[float],
    top_k: int,
    tree_root: QDTreeNode,
    partition_prefix: str,
) -> Tuple[List[Tuple[int, int, Any, float]], float]:
    user_roles = _fetch_user_roles(user_id)
    if not user_roles:
        logger.debug("No roles found for user_id=%s", user_id)
        return [], 0.0

    query_vec, query_param = _prepare_query_vector(query_vector)
    centroid_partitions = _collect_relevant_partitions(tree_root, user_roles, query_vec)
    centroid_with_access = [
        partition
        for partition in centroid_partitions
        if partition_has_accessible_documents(partition, user_roles)
    ]
    role_partitions = gather_role_accessible_partitions(tree_root, user_roles)
    partitions_dict: Dict[str, QDTreeNode] = {}
    for partition in centroid_with_access:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict[key] = partition
    for partition in role_partitions:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict.setdefault(key, partition)
    selected_partitions = list(partitions_dict.values())
    if centroid_with_access:
        logger.debug(
            "Centroid partitions with accessible docs: %s",
            [p.table_name or p.partition_id for p in centroid_with_access],
        )
    else:
        logger.debug("Centroid partitions had no accessible docs.")
    logger.debug(
        "Role-accessible partitions candidate: %s",
        [p.table_name or p.partition_id for p in role_partitions],
    )
    logger.debug(
        "Final partition list (order preserved): %s",
        [p.table_name or p.partition_id for p in selected_partitions],
    )
    partitions = selected_partitions

    if not partitions:
        logger.debug("No partitions matched for user_id=%s", user_id)
        return [], 0.0

    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    all_results: List[Tuple[int, int, Any, float]] = []
    start_time = time.time()

    try:
        import efconfig
        cur.execute("SET max_parallel_workers_per_gather = 0;")
        cur.execute("SET jit = off;")
        cur.execute("SET enable_seqscan = off;")
        cur.execute(f"SET hnsw.ef_search = {getattr(efconfig, 'ef_search', 40)};")
        logger.debug(
            "qd-tree system search configured: partitions=%s",
            [p.table_name or p.partition_id for p in partitions],
        )

        for partition in partitions:
            table_name = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
            if table_name is None:
                continue

            doc_id_filter = sorted(_collect_partition_document_ids_for_user(partition, user_roles))
            if not doc_id_filter:
                logger.debug(
                    "Skipping partition %s; no documents remain after role filtering",
                    table_name,
                )
                continue

            cur.execute(
                sql.SQL(
                    """
                    SELECT block_id, document_id, block_content,
                           vector <-> %s::vector AS distance
                    FROM {}
                    WHERE document_id = ANY(%s)
                    ORDER BY distance
                    LIMIT %s;
                    """
                ).format(sql.Identifier(table_name)),
                [query_param, doc_id_filter, top_k],
            )
            partition_rows = cur.fetchall()
            all_results.extend(partition_rows)
    finally:
        cur.close()
        conn.close()

    elapsed = time.time() - start_time
    filtered_results = _merge_qd_tree_results(all_results, top_k)
    return filtered_results, elapsed


def qd_tree_search(
    user_id: str,
    query_vector: Sequence[float],
    topk: int = 5,
    statistics_type: str = "sql",
    tree_path: Optional[str] = None,
    partition_prefix: Optional[str] = None,
) -> Tuple[List[Tuple[int, int, Any, float]], float]:
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    root = get_qd_tree_root(tree_path=tree_path, partition_prefix=prefix)
    if statistics_type == "sql":
        return qd_tree_search_statistics_sql(
            user_id=user_id,
            query_vector=query_vector,
            top_k=topk,
            tree_root=root,
            partition_prefix=prefix,
        )
    elif statistics_type == "system":
        return qd_tree_search_statistics_system(
            user_id=user_id,
            query_vector=query_vector,
            top_k=topk,
            tree_root=root,
            partition_prefix=prefix,
        )
    else:
        raise ValueError(f"Unsupported statistics_type '{statistics_type}'")


def execute_hq_query_sql(
    user_id: str,
    query_vector: Sequence[float],
    rbac_policy: Mapping[str, Iterable[str]],
    qd_tree_root: QDTreeNode,
    top_k: int = 5,
    partition_prefix: str = "documentblocks_qdtree_partition",
) -> List[Tuple[int, int]]:
    results, _ = qd_tree_search_statistics_sql(
        user_id=user_id,
        query_vector=query_vector,
        top_k=top_k,
        tree_root=qd_tree_root,
        partition_prefix=partition_prefix,
    )
    return [(doc_id, block_id) for block_id, doc_id, _content, _distance in results]


def _list_qdtree_partition_tables(partition_prefix: str) -> List[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE %s;
            """,
            (f"{partition_prefix}_%",),
        )
        tables = [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()
    return tables


def create_index_for_qdtree_partition(table_name: str, index_type: str = "hnsw") -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        _create_partition_indexes(cur, table_name, index_type=index_type)
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Error creating index for {table_name}: {exc}") from exc
    finally:
        cur.close()
        conn.close()


def create_indexes_for_qdtree_partitions(
    index_type: str = "hnsw",
    partition_prefix: Optional[str] = None,
    workers: Optional[int] = None,
) -> None:
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    tables = _list_qdtree_partition_tables(prefix)
    if not tables:
        logger.info("No QD-tree partition tables found with prefix '%s'", prefix)
        return

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        maintenance_settings = get_maintenance_settings()
        maintenance_work_mem_gb = maintenance_settings["maintenance_work_mem_gb"]
        max_parallel_workers = maintenance_settings["max_parallel_maintenance_workers"]
        cur.execute(f"SET maintenance_work_mem = '{maintenance_work_mem_gb}GB';")
        cur.execute(f"SET max_parallel_maintenance_workers = {max_parallel_workers};")
        conn.commit()
    except psycopg2.Error:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    worker_count = workers or os.cpu_count() or 1
    logger.info(
        "Building %s indexes for %s tables using %s worker(s)",
        index_type,
        len(tables),
        worker_count,
    )
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(create_index_for_qdtree_partition, table, index_type) for table in tables]
        for future in futures:
            future.result()


def drop_indexes_for_qdtree_partitions(partition_prefix: Optional[str] = None) -> None:
    prefix = partition_prefix or DEFAULT_QD_TREE_PARTITION_PREFIX
    tables = _list_qdtree_partition_tables(prefix)
    if not tables:
        logger.info("No QD-tree partition tables found with prefix '%s'", prefix)
        return

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for table_name in tables:
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(
                    sql.Identifier(f"{table_name}_vector_idx")
                )
            )
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Error dropping indexes for prefix '{prefix}': {exc}") from exc
    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_qd_tree(root: QDTreeNode, path: str) -> None:
    output_path = Path(path).expanduser().resolve()
    logger.info("Saving QD-tree to '%s'", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "wb") as fout:
            pickle.dump(root, fout)
    except OSError as exc:
        logger.error(
            "Failed to save QD-tree to '%s': %s. Consider specifying a different "
            "location with --output-dir or --output.",
            output_path,
            exc,
        )
        raise


def load_qd_tree(path: str) -> QDTreeNode:
    logger.info("Loading QD-tree from '%s'", path)
    with open(path, "rb") as fin:
        root: QDTreeNode = pickle.load(fin)
    return root
