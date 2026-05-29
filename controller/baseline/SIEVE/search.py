from __future__ import annotations

import importlib
import sys
import time
from collections import deque
from typing import Optional

from psycopg2 import sql

from services.config import get_db_connection

from .common import SIEVE_ROOT_TABLE, SievePartition
from .cost_model import SieveCostModel
from .storage import load_current_hasse_children, get_current_plan_summary, load_current_partitions, load_current_user_roles


def _resolve_efconfig_module():
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        module = sys.modules.get(module_name)
        if module is not None:
            return module
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _configured_int(primary_name: str, default: int, *, fallback_name: str | None = None) -> int:
    efconfig = _resolve_efconfig_module()
    value = None
    if efconfig is not None and hasattr(efconfig, primary_name):
        value = getattr(efconfig, primary_name)
    elif fallback_name and efconfig is not None and hasattr(efconfig, fallback_name):
        value = getattr(efconfig, fallback_name)
    if value is None:
        return max(1, int(default))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "adaptive", "auto", "none"}:
            return max(1, int(default))
        return max(1, int(float(normalized)))
    return max(1, int(value))


def _parse_vector(query_vector):
    if isinstance(query_vector, str):
        return query_vector
    if hasattr(query_vector, "tolist"):
        return str(list(query_vector.tolist()))
    return str(list(query_vector))


def _normalize_roles(role_ids) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in (role_ids or ())}))


def _is_subset(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> bool:
    return set(lhs).issubset(set(rhs))


def _merge_results(all_results, topk: int):
    seen = set()
    unique_results = []
    all_results.sort(key=lambda row: row[3])
    for row in all_results:
        key = (row[1], row[0])
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(row)
        if len(unique_results) == int(topk):
            break
    return unique_results


def _build_cost_model(summary: dict[str, object]) -> SieveCostModel:
    metadata = dict(summary.get("metadata", {}))
    return SieveCostModel(
        dataset_size=int(summary.get("dataset_size", 1)),
        m=int(metadata.get("m", 16)),
        bitvector_cutoff=int(metadata.get("bitvector_cutoff", 1000)),
        ef_search=int(metadata.get("ef_search", 10)),
        k=10,
        heterogeneous_indexing=bool(metadata.get("heterogeneous_indexing", True)),
        heterogeneous_search=bool(metadata.get("heterogeneous_search", True)),
    )


def _base_ef(summary: dict[str, object]) -> int:
    metadata = dict(summary.get("metadata", {}))
    return _configured_int("sieve_ef_search", int(metadata.get("ef_search", 10)), fallback_name="ef_search")


def _best_ancestor_partition(
    role_ids: tuple[int, ...],
    partitions: list[SievePartition],
    *,
    root_table: str,
    hasse_children: dict[str, tuple[str, ...]],
) -> Optional[SievePartition]:
    partition_by_id = {partition.partition_id: partition for partition in partitions}
    partition_by_role_ids = {_normalize_roles(partition.role_ids): partition for partition in partitions}

    exact_partition = partition_by_role_ids.get(role_ids)
    if exact_partition is not None:
        return exact_partition

    best = None
    best_cardinality = None
    visited: set[str] = set()
    queue = deque(hasse_children.get(root_table, ()))

    while queue:
        partition_id = queue.popleft()
        if partition_id in visited:
            continue
        visited.add(partition_id)

        partition = partition_by_id.get(partition_id)
        if partition is None:
            continue

        partition_roles = _normalize_roles(partition.role_ids)
        if not _is_subset(role_ids, partition_roles):
            continue

        if best is None or int(partition.cardinality) < int(best_cardinality):
            best = partition
            best_cardinality = int(partition.cardinality)

        for child_id in hasse_children.get(partition_id, ()):
            if child_id not in visited:
                queue.append(child_id)

    return best


def _prepare_to_cover_table(cur, *, root_table: str, role_ids: tuple[int, ...]) -> None:
    temp_table = sql.Identifier("sieve_query_to_cover")
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(temp_table))
    cur.execute(
        sql.SQL(
            """
            CREATE TEMP TABLE {} (
                block_id BIGINT NOT NULL,
                document_id INT NOT NULL,
                PRIMARY KEY (block_id, document_id)
            ) ON COMMIT DROP;
            """
        ).format(temp_table)
    )
    cur.execute(
        sql.SQL(
            """
            INSERT INTO {} (block_id, document_id)
            SELECT block_id, document_id
            FROM {}
            WHERE role_ids && %s::BIGINT[];
            """
        ).format(temp_table, sql.Identifier(root_table)),
        [list(role_ids)],
    )
    cur.execute(sql.SQL("ANALYZE {};").format(temp_table))


def _remaining_to_cover(cur) -> int:
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier("sieve_query_to_cover")))
    return int(cur.fetchone()[0] or 0)


def _partition_to_cover_intersection(cur, table_name: str) -> int:
    cur.execute(
        sql.SQL(
            """
            SELECT COUNT(*)
            FROM {} q
            JOIN {} p ON p.block_id = q.block_id AND p.document_id = q.document_id;
            """
        ).format(sql.Identifier("sieve_query_to_cover"), sql.Identifier(table_name))
    )
    return int(cur.fetchone()[0] or 0)


def _partition_query_intersection(cur, table_name: str, role_ids: tuple[int, ...]) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {} WHERE role_ids && %s::BIGINT[];").format(sql.Identifier(table_name)),
        [list(role_ids)],
    )
    return int(cur.fetchone()[0] or 0)


def _delete_partition_from_to_cover(cur, table_name: str) -> None:
    cur.execute(
        sql.SQL(
            """
            DELETE FROM {} q
            USING {} p
            WHERE p.block_id = q.block_id AND p.document_id = q.document_id;
            """
        ).format(sql.Identifier("sieve_query_to_cover"), sql.Identifier(table_name))
    )


def _covering_partitions_sql(
    cur,
    *,
    root_table: str,
    role_ids: tuple[int, ...],
    partitions: list[SievePartition],
    cost_model: SieveCostModel,
    query_cardinality: int,
) -> tuple[list[SievePartition], float]:
    if not partitions:
        return [], float("inf")

    _prepare_to_cover_table(cur, root_table=root_table, role_ids=role_ids)
    selected: list[SievePartition] = []
    already_selected: set[str] = set()
    query_intersection_cache: dict[str, int] = {}
    total_cost = 0.0

    while _remaining_to_cover(cur) > 0:
        best_partition = None
        best_ratio = float("inf")
        best_search_cost = 0.0

        for partition in partitions:
            if partition.partition_id in already_selected:
                continue
            partition_roles = _normalize_roles(partition.role_ids)
            if _is_subset(role_ids, partition_roles):
                already_selected.add(partition.partition_id)
                continue

            intersect_size = _partition_to_cover_intersection(cur, partition.table_name)
            if intersect_size <= 0:
                continue

            if partition.partition_id not in query_intersection_cache:
                query_intersection_cache[partition.partition_id] = _partition_query_intersection(
                    cur,
                    partition.table_name,
                    role_ids,
                )
            intersect_selectivity = max(1, int(query_intersection_cache[partition.partition_id]))
            search_cost = cost_model.upward_search_cost(int(partition.cardinality), intersect_selectivity)
            ratio = search_cost / float(intersect_size)
            if ratio < best_ratio:
                best_ratio = ratio
                best_search_cost = search_cost
                best_partition = partition

        if best_partition is None:
            return [], float("inf")

        selected.append(best_partition)
        already_selected.add(best_partition.partition_id)
        total_cost += best_search_cost
        _delete_partition_from_to_cover(cur, best_partition.table_name)

    if _remaining_to_cover(cur) > 0:
        return [], float("inf")
    return selected, total_cost


def _extract_execution_time(explain_rows) -> float:
    total = 0.0
    for (line,) in explain_rows:
        if "Execution Time" in line:
            total += float(line.split()[-2]) / 1000.0
    return total


def _set_index_usage(cur, *, force_seqscan: bool) -> None:
    if force_seqscan:
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET enable_indexonlyscan = off;")
    else:
        cur.execute("RESET enable_indexscan;")
        cur.execute("RESET enable_bitmapscan;")
        cur.execute("RESET enable_indexonlyscan;")


def _set_search_session(cur, *, ef_search: int, force_seqscan: bool = False) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")
    _set_index_usage(cur, force_seqscan=force_seqscan)
    if not force_seqscan:
        cur.execute(f"SET hnsw.ef_search = {max(1, int(ef_search))};")


def _run_table_search(
    cur,
    *,
    table_name: str,
    query_vector,
    role_ids: tuple[int, ...],
    topk: int,
    ef_search: int,
    force_seqscan: bool,
    collect_sql_time: bool,
) -> tuple[list[tuple], float]:
    _set_search_session(cur, ef_search=ef_search, force_seqscan=force_seqscan)
    query = sql.SQL(
        """
        SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
        FROM {}
        WHERE role_ids && %s::BIGINT[]
        ORDER BY distance
        LIMIT %s;
        """
    ).format(sql.Identifier(table_name))

    sql_time = 0.0
    if collect_sql_time:
        cur.execute(sql.SQL("EXPLAIN ANALYZE ") + query, [query_vector, list(role_ids), int(topk)])
        sql_time += _extract_execution_time(cur.fetchall())
    cur.execute(query, [query_vector, list(role_ids), int(topk)])
    return cur.fetchall(), sql_time


def _sieve_search_impl(user_id: int, query_vector, topk: int, *, collect_sql_time: bool):
    summary = get_current_plan_summary(refresh=False)
    if summary is None:
        return [], 0.0

    root_table = str(summary.get("root_table_name") or SIEVE_ROOT_TABLE)
    partitions = load_current_partitions(refresh=False)
    hasse_children = load_current_hasse_children(refresh=False)
    cost_model = _build_cost_model(summary)
    base_ef = _base_ef(summary)
    query_vector_sql = _parse_vector(query_vector)
    enable_multipartition_search = bool(dict(summary.get("metadata", {})).get("enable_multipartition_search", False))

    conn = get_db_connection()
    started_at = time.time()
    sql_time = 0.0
    try:
        with conn.cursor() as cur:
            user_roles = load_current_user_roles(refresh=False)
            role_ids = tuple(int(role_id) for role_id in user_roles.get(int(user_id), ()))
            if not role_ids:
                return [], 0.0

            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {} WHERE role_ids && %s::BIGINT[];").format(sql.Identifier(root_table)),
                [list(role_ids)],
            )
            query_cardinality = int(cur.fetchone()[0] or 0)
            if query_cardinality <= 0:
                return [], 0.0

            best_partition = _best_ancestor_partition(
                role_ids,
                partitions,
                root_table=root_table,
                hasse_children=hasse_children,
            )
            best_table = root_table
            best_cardinality = int(summary.get("dataset_size", 1))
            if best_partition is not None:
                best_table = best_partition.table_name
                best_cardinality = int(best_partition.cardinality)

            upward_cost = cost_model.upward_search_cost(max(1, int(best_cardinality)), max(1, int(query_cardinality)))
            bf_cost = cost_model.bf_search_cost(query_cardinality)
            if query_cardinality <= cost_model.bitvector_cutoff:
                strategy = "bruteforce"
                covering = []
                covering_cost = float("inf")
            else:
                covering = []
                covering_cost = float("inf")
                if enable_multipartition_search:
                    covering, covering_cost = _covering_partitions_sql(
                        cur,
                        root_table=root_table,
                        role_ids=role_ids,
                        partitions=partitions,
                        cost_model=cost_model,
                        query_cardinality=query_cardinality,
                    )
                strategy = "upward"
                if bf_cost <= upward_cost and bf_cost <= covering_cost:
                    strategy = "bruteforce"
                elif enable_multipartition_search and covering_cost <= bf_cost and covering_cost <= upward_cost:
                    strategy = "covering"

            if strategy == "bruteforce":
                results, elapsed = _run_table_search(
                    cur,
                    table_name=root_table,
                    query_vector=query_vector_sql,
                    role_ids=role_ids,
                    topk=int(topk),
                    ef_search=base_ef,
                    force_seqscan=True,
                    collect_sql_time=collect_sql_time,
                )
                sql_time += elapsed
            elif strategy == "covering":
                collected = []
                for partition in covering:
                    partition_ef = cost_model.downscaled_ef_search(
                        int(partition.cardinality),
                        ef_search=base_ef,
                        k=int(topk),
                    )
                    rows, elapsed = _run_table_search(
                        cur,
                        table_name=partition.table_name,
                        query_vector=query_vector_sql,
                        role_ids=role_ids,
                        topk=int(topk),
                        ef_search=partition_ef,
                        force_seqscan=False,
                        collect_sql_time=collect_sql_time,
                    )
                    sql_time += elapsed
                    collected.extend(rows)
                results = _merge_results(collected, int(topk))
            else:
                partition_ef = cost_model.downscaled_ef_search(
                    int(best_cardinality),
                    ef_search=base_ef,
                    k=int(topk),
                )
                results, elapsed = _run_table_search(
                    cur,
                    table_name=best_table,
                    query_vector=query_vector_sql,
                    role_ids=role_ids,
                    topk=int(topk),
                    ef_search=partition_ef,
                    force_seqscan=False,
                    collect_sql_time=collect_sql_time,
                )
                sql_time += elapsed
    finally:
        conn.close()

    if collect_sql_time:
        return results, sql_time
    return results, time.time() - started_at


def sieve_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return sieve_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return sieve_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def sieve_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    return _sieve_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=True)


def sieve_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    return _sieve_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=False)
