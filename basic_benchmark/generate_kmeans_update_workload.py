from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.config import get_db_connection
from controller.kmeans.common import PARTITION_TABLE


@dataclass(frozen=True, slots=True)
class DocumentState:
    document_id: int
    role_ids: tuple[int, ...]
    tenant_ids: tuple[int, ...]
    vector_count: int


INSERT_WEIGHTS = {
    "insert_existing_acl": 2.0,
    "insert_acl_union": 1.0,
    "insert_acl_sample": 0.0,
}

UPDATE_WEIGHTS = {
    "acl_existing": 1.0,
    "acl_widen": 1.0,
    "vector_update": 1.0,
    "acl_narrow": 0.0,
    "acl_clear": 0.0,
}

DELETE_WEIGHTS = {
    "delete_random": 1.0,
    "delete_hot_pattern": 1.0,
    "delete_recent": 1.0,
}


def _str_to_bool(value: str) -> bool:
    lowered = str(value).lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in weights}
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
    if not positive:
        raise ValueError("at least one weight must be positive")
    weight_sum = sum(positive.values())
    raw = {key: float(total) * value / weight_sum for key, value in positive.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = int(total) - sum(result.values())
    ranked = sorted(positive, key=lambda key: (-(raw[key] - result[key]), key))
    for key in ranked[:remainder]:
        result[key] += 1
    for key in weights:
        result.setdefault(key, 0)
    return result


def _tenant_key(tenant_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in tenant_ids}))


def _fetch_documents(*, max_document_id: int | None = None) -> dict[int, DocumentState]:
    where = ""
    params: list[int] = []
    if max_document_id is not None:
        where = "WHERE db.document_id <= %s"
        params.append(int(max_document_id))
    query = f"""
        WITH doc_blocks AS (
            SELECT document_id, COUNT(*)::BIGINT AS vector_count
            FROM documentblocks
            GROUP BY document_id
        ),
        doc_acl AS (
            SELECT pa.document_id,
                   array_agg(DISTINCT pa.role_id ORDER BY pa.role_id) AS role_ids,
                   array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON ur.role_id = pa.role_id
            GROUP BY pa.document_id
        )
        SELECT db.document_id,
               COALESCE(acl.role_ids, '{{}}') AS role_ids,
               COALESCE(acl.tenant_ids, '{{}}') AS tenant_ids,
               db.vector_count
        FROM doc_blocks db
        LEFT JOIN doc_acl acl ON acl.document_id = db.document_id
        {where}
        ORDER BY db.document_id;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        int(document_id): DocumentState(
            document_id=int(document_id),
            role_ids=tuple(int(value) for value in (role_ids or ())),
            tenant_ids=tuple(int(value) for value in (tenant_ids or ())),
            vector_count=int(vector_count),
        )
        for document_id, role_ids, tenant_ids, vector_count in rows
        if int(vector_count) > 0
    }


def _fetch_all_tenants() -> tuple[int, ...]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM UserRoles ORDER BY user_id;")
            rows = cur.fetchall()
    finally:
        conn.close()
    return tuple(int(row[0]) for row in rows)


def _fetch_next_document_id() -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(document_id), 0) + 1
                FROM (
                    SELECT document_id FROM documentblocks
                    UNION
                    SELECT document_id FROM PermissionAssignment
                    UNION
                    SELECT document_id FROM Documents
                ) AS docs;
                """
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return int(row[0])


def _fetch_kmeans_partition_documents() -> dict[str, tuple[int, ...]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s);", [PARTITION_TABLE])
            if cur.fetchone()[0] is None:
                return {}
            cur.execute(
                f"""
                SELECT partition_id, document_ids
                FROM {PARTITION_TABLE}
                ORDER BY partition_id;
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        str(partition_id): tuple(sorted({int(value) for value in (document_ids or ())}))
        for partition_id, document_ids in rows
    }


def _choose_doc(
    rng: random.Random,
    docs: dict[int, DocumentState],
    *,
    excluded: set[int],
    require_acl: bool = False,
    require_multi_tenant: bool = False,
    candidates: Iterable[int] | None = None,
) -> DocumentState | None:
    if candidates is None:
        pool = list(docs.values())
    else:
        pool = [docs[int(document_id)] for document_id in candidates if int(document_id) in docs]
    pool = [
        doc
        for doc in pool
        if int(doc.document_id) not in excluded
        and (not require_acl or bool(doc.tenant_ids or doc.role_ids))
        and (not require_multi_tenant or len(doc.tenant_ids) > 1)
    ]
    if not pool:
        return None
    return rng.choice(pool)


def _random_tenant_sample(
    rng: random.Random,
    all_tenants: tuple[int, ...],
    *,
    min_size: int,
    max_size: int,
) -> tuple[int, ...]:
    if not all_tenants:
        raise RuntimeError("No tenants found in UserRoles")
    upper = min(int(max_size), len(all_tenants))
    lower = min(max(1, int(min_size)), upper)
    size = rng.randint(lower, upper)
    return tuple(sorted(rng.sample(list(all_tenants), size)))


def _unique_tenant_set(
    rng: random.Random,
    base: Iterable[int],
    *,
    all_tenants: tuple[int, ...],
    known_acl_keys: set[tuple[int, ...]],
    generated_acl_keys: set[tuple[int, ...]],
    min_size: int,
    max_size: int,
) -> tuple[int, ...]:
    base_set = set(int(value) for value in base)
    if not base_set:
        base_set.update(_random_tenant_sample(rng, all_tenants, min_size=min_size, max_size=max_size))
    for _ in range(128):
        candidate = set(base_set)
        key = _tenant_key(candidate)
        if key and key not in known_acl_keys and key not in generated_acl_keys:
            generated_acl_keys.add(key)
            return key
        missing = [tenant_id for tenant_id in all_tenants if int(tenant_id) not in candidate]
        if missing:
            candidate.add(int(rng.choice(missing)))
        elif len(candidate) > 1:
            candidate.remove(int(rng.choice(sorted(candidate))))
        base_set = candidate
    for _ in range(512):
        candidate = _random_tenant_sample(rng, all_tenants, min_size=min_size, max_size=max_size)
        key = _tenant_key(candidate)
        if key not in known_acl_keys and key not in generated_acl_keys:
            generated_acl_keys.add(key)
            return key
    key = _tenant_key(base_set)
    generated_acl_keys.add(key)
    return key


def _proper_subset(
    rng: random.Random,
    tenant_ids: tuple[int, ...],
    *,
    known_acl_keys: set[tuple[int, ...]],
    generated_acl_keys: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    tenants = tuple(sorted({int(value) for value in tenant_ids}))
    if len(tenants) <= 1:
        return None
    for _ in range(128):
        size = rng.randint(1, len(tenants) - 1)
        candidate = tuple(sorted(rng.sample(list(tenants), size)))
        if candidate not in known_acl_keys and candidate not in generated_acl_keys:
            generated_acl_keys.add(candidate)
            return candidate
    size = max(1, len(tenants) // 2)
    return tuple(sorted(rng.sample(list(tenants), size)))


def _hot_pattern_candidates(docs: dict[int, DocumentState]) -> list[int]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for doc in docs.values():
        key = _tenant_key(doc.tenant_ids)
        if key:
            groups[key].append(int(doc.document_id))
    ranked_groups = sorted(groups.values(), key=lambda values: (-len(values), min(values)))
    result: list[int] = []
    for group in ranked_groups[: max(1, min(16, len(ranked_groups)))]:
        result.extend(sorted(group))
    return result


def _add_acl_payload(item: dict[str, object], source: DocumentState) -> None:
    if source.role_ids:
        item["role_ids"] = list(source.role_ids)
    elif source.tenant_ids:
        item["tenant_ids"] = list(source.tenant_ids)


def generate_workload(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(int(args.seed))
    total = int(args.total) if args.total is not None else int(args.batch_size) * int(args.batches)
    if total <= 0:
        raise ValueError("--total must be positive")
    batch_size = int(args.batch_size)
    batches = int(math.ceil(float(total) / float(batch_size)))

    active_docs = _fetch_documents()
    if not active_docs:
        raise RuntimeError("No documents with vectors found. Load the dataset first.")
    source_docs = _fetch_documents(max_document_id=args.source_document_max_id)
    if not source_docs:
        raise RuntimeError("No source documents found for vector copy.")

    all_tenants = _fetch_all_tenants()
    known_acl_keys = {_tenant_key(doc.tenant_ids) for doc in active_docs.values() if doc.tenant_ids}
    generated_acl_keys: set[tuple[int, ...]] = set()
    next_document_id = int(args.next_document_id or _fetch_next_document_id())

    workload: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    generated_doc_ids: set[int] = set()
    batch_locality_details: list[dict[str, object]] = []
    partition_documents = _fetch_kmeans_partition_documents() if int(args.batch_locality_partitions) > 0 else {}
    partition_ids_by_document: dict[int, set[str]] = defaultdict(set)
    for partition_id, document_ids in partition_documents.items():
        for document_id in document_ids:
            partition_ids_by_document[int(document_id)].add(str(partition_id))

    def choose_batch_locality(batch_index: int) -> tuple[set[int] | None, dict[str, object] | None]:
        if int(args.batch_locality_partitions) <= 0:
            return None, None
        eligible = [
            (str(partition_id), [int(doc_id) for doc_id in document_ids if int(doc_id) in active_docs])
            for partition_id, document_ids in partition_documents.items()
        ]
        eligible = [(partition_id, docs) for partition_id, docs in eligible if docs]
        if not eligible:
            if bool(args.batch_locality_strict):
                raise RuntimeError("No active documents found in current kmeans partitions for local workload batching")
            return None, {
                "batch_index": int(batch_index),
                "anchor_partition_ids": [],
                "anchor_document_count": 0,
                "fallback_global": True,
            }
        selected: list[tuple[str, list[int]]] = []
        available = list(eligible)
        for _ in range(min(int(args.batch_locality_partitions), len(available))):
            total_docs = sum(len(docs) for _partition_id, docs in available)
            ticket = rng.randrange(max(1, int(total_docs)))
            cursor = 0
            selected_index = 0
            for index, (_partition_id, docs) in enumerate(available):
                cursor += len(docs)
                if ticket < cursor:
                    selected_index = index
                    break
            selected.append(available.pop(selected_index))
        preferred = {int(doc_id) for _partition_id, docs in selected for doc_id in docs}
        return preferred, {
            "batch_index": int(batch_index),
            "anchor_partition_ids": [str(partition_id) for partition_id, _docs in selected],
            "anchor_document_count": int(len(preferred)),
            "fallback_global": False,
        }

    def vector_source(excluded: set[int], preferred_candidates: Iterable[int] | None = None) -> DocumentState | None:
        if preferred_candidates is not None:
            candidates = [doc_id for doc_id in preferred_candidates if doc_id in source_docs and doc_id in active_docs]
            doc = _choose_doc(rng, active_docs, excluded=excluded, candidates=candidates)
            if doc is not None:
                return doc
        candidates = [doc_id for doc_id in source_docs if doc_id in active_docs]
        doc = _choose_doc(rng, active_docs, excluded=excluded, candidates=candidates)
        if doc is not None:
            return doc
        return _choose_doc(rng, active_docs, excluded=excluded)

    def acl_source(excluded: set[int], preferred_candidates: Iterable[int] | None = None) -> DocumentState | None:
        if preferred_candidates is not None:
            candidates = [doc_id for doc_id in preferred_candidates if doc_id in source_docs and doc_id in active_docs]
            doc = _choose_doc(rng, active_docs, excluded=excluded, require_acl=True, candidates=candidates)
            if doc is not None:
                return doc
        candidates = [doc_id for doc_id in source_docs if doc_id in active_docs]
        doc = _choose_doc(rng, active_docs, excluded=excluded, require_acl=True, candidates=candidates)
        if doc is not None:
            return doc
        return _choose_doc(rng, active_docs, excluded=excluded, require_acl=True)

    def append_insert(
        subtype: str,
        batch_index: int,
        touched: set[int],
        protected_sources: set[int],
        preferred_candidates: Iterable[int] | None,
    ) -> bool:
        nonlocal next_document_id
        source = vector_source(excluded=set(), preferred_candidates=preferred_candidates)
        if source is None:
            return False
        protected_sources.add(int(source.document_id))
        document_id = int(next_document_id)
        next_document_id += 1
        item: dict[str, object] = {
            "operation": "insert",
            "document_id": document_id,
            "source_document_id": int(source.document_id),
            "document_name": f"kmeans_update_{subtype}_{document_id}",
            "subtype": subtype,
            "batch_index": int(batch_index),
            "source_vector_count": int(source.vector_count),
        }
        if subtype == "insert_existing_acl":
            acl_doc = acl_source(excluded=set(), preferred_candidates=preferred_candidates) or source
            if not (acl_doc.role_ids or acl_doc.tenant_ids):
                return False
            protected_sources.add(int(acl_doc.document_id))
            _add_acl_payload(item, acl_doc)
            tenant_ids = tuple(acl_doc.tenant_ids)
            role_ids = tuple(acl_doc.role_ids)
            item["source_acl_document_id"] = int(acl_doc.document_id)
            item["source_tenant_count"] = int(len(tenant_ids))
        elif subtype == "insert_acl_union":
            left = acl_source(excluded=set(), preferred_candidates=preferred_candidates)
            right = acl_source(
                excluded={int(left.document_id)} if left is not None else set(),
                preferred_candidates=preferred_candidates,
            )
            if left is None or right is None:
                return False
            protected_sources.update({int(left.document_id), int(right.document_id)})
            tenant_ids = _unique_tenant_set(
                rng,
                set(left.tenant_ids) | set(right.tenant_ids),
                all_tenants=all_tenants,
                known_acl_keys=known_acl_keys,
                generated_acl_keys=generated_acl_keys,
                min_size=int(args.min_synthetic_tenants),
                max_size=int(args.max_synthetic_tenants),
            )
            role_ids = ()
            item["tenant_ids"] = list(tenant_ids)
            item["left_acl_document_id"] = int(left.document_id)
            item["right_acl_document_id"] = int(right.document_id)
            item["new_tenant_count"] = int(len(tenant_ids))
        elif subtype == "insert_acl_sample":
            tenant_ids = _unique_tenant_set(
                rng,
                _random_tenant_sample(
                    rng,
                    all_tenants,
                    min_size=int(args.min_synthetic_tenants),
                    max_size=int(args.max_synthetic_tenants),
                ),
                all_tenants=all_tenants,
                known_acl_keys=known_acl_keys,
                generated_acl_keys=generated_acl_keys,
                min_size=int(args.min_synthetic_tenants),
                max_size=int(args.max_synthetic_tenants),
            )
            role_ids = ()
            item["tenant_ids"] = list(tenant_ids)
            item["new_tenant_count"] = int(len(tenant_ids))
        else:
            raise ValueError(f"unknown insert subtype: {subtype}")

        active_docs[document_id] = DocumentState(
            document_id=document_id,
            role_ids=role_ids,
            tenant_ids=tenant_ids,
            vector_count=int(source.vector_count),
        )
        touched.add(document_id)
        generated_doc_ids.add(document_id)
        workload.append(item)
        counts[("insert", subtype)] += 1
        if tenant_ids:
            known_acl_keys.add(_tenant_key(tenant_ids))
        return True

    def append_update(
        subtype: str,
        batch_index: int,
        touched: set[int],
        protected_sources: set[int],
        preferred_candidates: Iterable[int] | None,
    ) -> bool:
        target_excluded = set(touched)
        if subtype == "acl_existing":
            target = _choose_doc(
                rng,
                active_docs,
                excluded=target_excluded,
                require_acl=True,
                candidates=preferred_candidates,
            )
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=target_excluded, require_acl=True)
            donor = acl_source(
                excluded={int(target.document_id)} if target is not None else set(),
                preferred_candidates=preferred_candidates,
            )
            if target is None or donor is None or not (donor.role_ids or donor.tenant_ids):
                return False
            protected_sources.add(int(donor.document_id))
            item = {
                "operation": "acl_update",
                "document_id": int(target.document_id),
                "subtype": subtype,
                "batch_index": int(batch_index),
                "old_tenant_count": int(len(target.tenant_ids)),
                "new_tenant_count": int(len(donor.tenant_ids)),
                "donor_document_id": int(donor.document_id),
            }
            _add_acl_payload(item, donor)
            active_docs[int(target.document_id)] = DocumentState(
                int(target.document_id),
                tuple(donor.role_ids),
                tuple(donor.tenant_ids),
                int(target.vector_count),
            )
        elif subtype == "acl_widen":
            target = _choose_doc(
                rng,
                active_docs,
                excluded=target_excluded,
                require_acl=True,
                candidates=preferred_candidates,
            )
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=target_excluded, require_acl=True)
            donor = acl_source(
                excluded={int(target.document_id)} if target is not None else set(),
                preferred_candidates=preferred_candidates,
            )
            if target is None or donor is None:
                return False
            new_tenants = _unique_tenant_set(
                rng,
                set(target.tenant_ids) | set(donor.tenant_ids),
                all_tenants=all_tenants,
                known_acl_keys=known_acl_keys,
                generated_acl_keys=generated_acl_keys,
                min_size=int(args.min_synthetic_tenants),
                max_size=int(args.max_synthetic_tenants),
            )
            protected_sources.add(int(donor.document_id))
            item = {
                "operation": "acl_update",
                "document_id": int(target.document_id),
                "tenant_ids": list(new_tenants),
                "subtype": subtype,
                "batch_index": int(batch_index),
                "old_tenant_count": int(len(target.tenant_ids)),
                "new_tenant_count": int(len(new_tenants)),
                "donor_document_id": int(donor.document_id),
            }
            active_docs[int(target.document_id)] = DocumentState(int(target.document_id), (), new_tenants, int(target.vector_count))
            known_acl_keys.add(_tenant_key(new_tenants))
        elif subtype == "acl_narrow":
            target = _choose_doc(
                rng,
                active_docs,
                excluded=target_excluded,
                require_acl=True,
                require_multi_tenant=True,
                candidates=preferred_candidates,
            )
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=target_excluded, require_acl=True, require_multi_tenant=True)
            if target is None:
                return False
            new_tenants = _proper_subset(
                rng,
                target.tenant_ids,
                known_acl_keys=known_acl_keys,
                generated_acl_keys=generated_acl_keys,
            )
            if not new_tenants:
                return False
            item = {
                "operation": "acl_update",
                "document_id": int(target.document_id),
                "tenant_ids": list(new_tenants),
                "subtype": subtype,
                "batch_index": int(batch_index),
                "old_tenant_count": int(len(target.tenant_ids)),
                "new_tenant_count": int(len(new_tenants)),
            }
            active_docs[int(target.document_id)] = DocumentState(int(target.document_id), (), new_tenants, int(target.vector_count))
            known_acl_keys.add(_tenant_key(new_tenants))
        elif subtype == "acl_clear":
            target = _choose_doc(
                rng,
                active_docs,
                excluded=target_excluded,
                require_acl=True,
                candidates=preferred_candidates,
            )
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=target_excluded, require_acl=True)
            if target is None:
                return False
            item = {
                "operation": "acl_update",
                "document_id": int(target.document_id),
                "subtype": subtype,
                "batch_index": int(batch_index),
                "old_tenant_count": int(len(target.tenant_ids)),
                "new_tenant_count": 0,
            }
            active_docs[int(target.document_id)] = DocumentState(int(target.document_id), (), (), int(target.vector_count))
        elif subtype == "vector_update":
            target = _choose_doc(rng, active_docs, excluded=target_excluded, candidates=preferred_candidates)
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=target_excluded)
            source = vector_source(
                excluded={int(target.document_id)} if target is not None else set(),
                preferred_candidates=preferred_candidates,
            )
            if target is None or source is None:
                return False
            protected_sources.add(int(source.document_id))
            item = {
                "operation": "vector_update",
                "document_id": int(target.document_id),
                "source_document_id": int(source.document_id),
                "subtype": subtype,
                "batch_index": int(batch_index),
                "old_vector_count": int(target.vector_count),
                "new_vector_count": int(source.vector_count),
            }
            active_docs[int(target.document_id)] = DocumentState(
                int(target.document_id),
                tuple(target.role_ids),
                tuple(target.tenant_ids),
                int(source.vector_count),
            )
        else:
            raise ValueError(f"unknown update subtype: {subtype}")
        touched.add(int(item["document_id"]))
        workload.append(item)
        counts[("update", subtype)] += 1
        return True

    def append_delete(
        subtype: str,
        batch_index: int,
        touched: set[int],
        protected_sources: set[int],
        prior_generated_doc_ids: set[int],
        preferred_candidates: Iterable[int] | None,
    ) -> bool:
        excluded = set(touched) | set(protected_sources)
        if subtype == "delete_hot_pattern":
            hot_candidates = _hot_pattern_candidates(active_docs)
            if preferred_candidates is not None:
                preferred_hot = [doc_id for doc_id in hot_candidates if int(doc_id) in set(map(int, preferred_candidates))]
                target = _choose_doc(rng, active_docs, excluded=excluded, candidates=preferred_hot)
            else:
                target = None
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=excluded, candidates=hot_candidates)
        elif subtype == "delete_recent":
            if not bool(args.allow_delete_recent):
                return False
            target = _choose_doc(rng, active_docs, excluded=excluded, candidates=prior_generated_doc_ids)
        elif subtype == "delete_random":
            target = _choose_doc(rng, active_docs, excluded=excluded, candidates=preferred_candidates)
            if target is None:
                target = _choose_doc(rng, active_docs, excluded=excluded)
        else:
            raise ValueError(f"unknown delete subtype: {subtype}")
        if target is None:
            return False
        item = {
            "operation": "delete",
            "document_id": int(target.document_id),
            "subtype": subtype,
            "batch_index": int(batch_index),
            "old_tenant_count": int(len(target.tenant_ids)),
            "old_vector_count": int(target.vector_count),
        }
        touched.add(int(target.document_id))
        active_docs.pop(int(target.document_id), None)
        workload.append(item)
        counts[("delete", subtype)] += 1
        return True

    def append_with_fallback(
        kind: str,
        subtype: str,
        batch_index: int,
        touched: set[int],
        protected_sources: set[int],
        prior_generated: set[int],
        preferred_candidates: Iterable[int] | None,
    ) -> None:
        if kind == "insert":
            options = [subtype] + [value for value in INSERT_WEIGHTS if value != subtype]
            for option in options:
                if append_insert(option, batch_index, touched, protected_sources, preferred_candidates):
                    if option != subtype:
                        fallback_counts[(subtype, option)] += 1
                    return
        elif kind == "update":
            options = [subtype] + [value for value in UPDATE_WEIGHTS if value != subtype]
            for option in options:
                if append_update(option, batch_index, touched, protected_sources, preferred_candidates):
                    if option != subtype:
                        fallback_counts[(subtype, option)] += 1
                    return
        elif kind == "delete":
            options = [subtype] + [value for value in DELETE_WEIGHTS if value != subtype]
            for option in options:
                if append_delete(option, batch_index, touched, protected_sources, prior_generated, preferred_candidates):
                    if option != subtype:
                        fallback_counts[(subtype, option)] += 1
                    return
        raise RuntimeError(f"Unable to generate {kind}/{subtype}; not enough eligible documents")

    remaining = int(total)
    for batch_index in range(1, batches + 1):
        current_batch_size = min(batch_size, remaining)
        remaining -= current_batch_size
        major_counts = _allocate(
            current_batch_size,
            {
                "insert": float(args.insert_ratio),
                "update": float(args.update_ratio),
                "delete": float(args.delete_ratio),
            },
        )
        insert_counts = _allocate(major_counts["insert"], INSERT_WEIGHTS)
        update_counts = _allocate(major_counts["update"], UPDATE_WEIGHTS)
        delete_counts = _allocate(major_counts["delete"], DELETE_WEIGHTS)
        touched: set[int] = set()
        protected_sources: set[int] = set()
        prior_generated = set(generated_doc_ids)
        preferred_candidates, locality_detail = choose_batch_locality(batch_index)
        if locality_detail is not None:
            batch_locality_details.append(locality_detail)

        for subtype, count in insert_counts.items():
            for _ in range(int(count)):
                append_with_fallback("insert", subtype, batch_index, touched, protected_sources, prior_generated, preferred_candidates)
        for subtype, count in update_counts.items():
            for _ in range(int(count)):
                append_with_fallback("update", subtype, batch_index, touched, protected_sources, prior_generated, preferred_candidates)
        for subtype, count in delete_counts.items():
            for _ in range(int(count)):
                append_with_fallback("delete", subtype, batch_index, touched, protected_sources, prior_generated, preferred_candidates)

    op_counts = Counter(str(item["operation"]) for item in workload)
    estimated_batch_partitions: dict[int, set[str]] = defaultdict(set)
    for item in workload:
        related_document_ids = {
            int(value)
            for key in ("document_id", "source_document_id", "source_acl_document_id", "left_acl_document_id", "right_acl_document_id", "donor_document_id")
            if item.get(key) is not None
            for value in (item.get(key),)
        }
        item_partition_ids = sorted(
            {
                str(partition_id)
                for document_id in related_document_ids
                for partition_id in partition_ids_by_document.get(int(document_id), set())
            }
        )
        if item_partition_ids:
            item["locality_partition_ids"] = item_partition_ids
            estimated_batch_partitions[int(item["batch_index"])].update(item_partition_ids)
    subtype_counts = Counter(str(item["subtype"]) for item in workload)
    new_acl_subtypes = {"insert_acl_union", "insert_acl_sample", "acl_widen", "acl_narrow"}
    insert_count = int(op_counts.get("insert", 0))
    update_count = int(op_counts.get("acl_update", 0) + op_counts.get("vector_update", 0))
    insert_new_acl_count = int(sum(1 for item in workload if str(item.get("operation")) == "insert" and str(item.get("subtype")) in new_acl_subtypes))
    update_new_acl_count = int(sum(1 for item in workload if str(item.get("operation")) in {"acl_update", "vector_update"} and str(item.get("subtype")) in new_acl_subtypes))
    summary = {
        "total": int(len(workload)),
        "batch_size": int(batch_size),
        "batches": int(batches),
        "seed": int(args.seed),
        "start_document_count": int(len(_fetch_documents())),
        "start_next_document_id": int(args.next_document_id or _fetch_next_document_id()),
        "end_next_document_id": int(next_document_id),
        "operation_counts": dict(sorted(op_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "new_acl_counts": {
            "insert_new_acl": int(insert_new_acl_count),
            "insert_total": int(insert_count),
            "insert_new_acl_ratio": float(insert_new_acl_count) / float(max(1, insert_count)),
            "update_new_acl": int(update_new_acl_count),
            "update_total": int(update_count),
            "update_new_acl_ratio": float(update_new_acl_count) / float(max(1, update_count)),
        },
        "fallback_counts": {f"{src}->{dst}": int(count) for (src, dst), count in sorted(fallback_counts.items())},
        "ratios": {
            "insert": float(args.insert_ratio),
            "update": float(args.update_ratio),
            "delete": float(args.delete_ratio),
        },
        "insert_subtypes": dict(INSERT_WEIGHTS),
        "update_subtypes": dict(UPDATE_WEIGHTS),
        "delete_subtypes": dict(DELETE_WEIGHTS),
        "source_document_max_id": None if args.source_document_max_id is None else int(args.source_document_max_id),
        "allow_delete_recent": bool(args.allow_delete_recent),
        "batch_locality_partitions": int(args.batch_locality_partitions),
        "batch_locality_strict": bool(args.batch_locality_strict),
        "batch_locality_details": batch_locality_details,
        "estimated_batch_partition_counts": {
            str(batch_index): int(len(partition_ids))
            for batch_index, partition_ids in sorted(estimated_batch_partitions.items())
        },
    }
    return workload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed kmeans update workload.")
    parser.add_argument("--output", default=str(Path(PROJECT_ROOT) / "basic_benchmark" / "kmeans_update_workload_mixed.json"))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--next-document-id", type=int, default=None)
    parser.add_argument("--source-document-max-id", type=int, default=None)
    parser.add_argument("--insert-ratio", type=float, default=0.5)
    parser.add_argument("--update-ratio", type=float, default=0.25)
    parser.add_argument("--delete-ratio", type=float, default=0.25)
    parser.add_argument("--min-synthetic-tenants", type=int, default=8)
    parser.add_argument("--max-synthetic-tenants", type=int, default=64)
    parser.add_argument("--allow-delete-recent", type=_str_to_bool, default=True)
    parser.add_argument(
        "--batch-locality-partitions",
        type=int,
        default=0,
        help="If positive, each generated batch prefers documents from this many current kmeans partitions.",
    )
    parser.add_argument(
        "--batch-locality-strict",
        type=_str_to_bool,
        default=False,
        help="Fail instead of falling back to global sampling when no local kmeans documents are available.",
    )
    parser.add_argument("--dry-run", type=_str_to_bool, default=False)
    args = parser.parse_args()

    workload, summary = generate_workload(args)
    print(json.dumps(summary, indent=2), flush=True)
    if bool(args.dry_run):
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(workload, file, indent=2)

    if args.summary_output:
        summary_path = Path(args.summary_output)
    else:
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"[kmeans-workload] wrote {len(workload)} updates to {output_path}", flush=True)
    print(f"[kmeans-workload] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
