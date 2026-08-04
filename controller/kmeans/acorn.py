from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable, Optional

from .common import PROJECT_ROOT


ACORN_BENCHMARK_DIR = Path(PROJECT_ROOT) / "acorn_benchmark"
ACORN_CONFIG_PATH = ACORN_BENCHMARK_DIR / "config.json"
KMEANS_ACORN_BINARY_CANDIDATES = (
    ACORN_BENCHMARK_DIR / "build" / "kmeans_acorn",
    ACORN_BENCHMARK_DIR / "kmeans_acorn",
)


def _binary_path() -> Path:
    for candidate in KMEANS_ACORN_BINARY_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    candidates = ", ".join(str(path) for path in KMEANS_ACORN_BINARY_CANDIDATES)
    raise RuntimeError(
        "KMeans ACORN helper binary is missing. Build it with "
        "`cmake -S acorn_benchmark -B acorn_benchmark/build && "
        "cmake --build acorn_benchmark/build --target kmeans_acorn`. "
        f"Looked for: {candidates}"
    )


def _index_storage_root() -> Path:
    if not ACORN_CONFIG_PATH.is_file():
        raise RuntimeError(f"ACORN config is missing: {ACORN_CONFIG_PATH}")
    payload = json.loads(ACORN_CONFIG_PATH.read_text(encoding="utf-8"))
    value = str(payload.get("index_storage_path") or "").strip()
    if not value:
        raise RuntimeError(f"`index_storage_path` is missing in {ACORN_CONFIG_PATH}")
    return Path(value)


def kmeans_acorn_index_dir() -> Path:
    return _index_storage_root() / "kmeans_partition"


def kmeans_acorn_index_path(table_name: str) -> Path:
    return kmeans_acorn_index_dir() / f"{table_name}.faiss"


def kmeans_acorn_index_metadata_path(table_name: str) -> Path:
    return kmeans_acorn_index_dir() / f"{table_name}.json"


def kmeans_acorn_index_exists(table_name: str) -> bool:
    return kmeans_acorn_index_path(str(table_name)).is_file()


def read_kmeans_acorn_index_metadata(table_name: str) -> Optional[dict[str, object]]:
    path = kmeans_acorn_index_metadata_path(str(table_name))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def kmeans_acorn_index_is_current(
    table_name: str,
    *,
    plan_id: Optional[int] = None,
    partition_id: Optional[str] = None,
    vector_count: Optional[int] = None,
    expected_index_kind: Optional[str] = None,
) -> bool:
    table_name = str(table_name)
    if not kmeans_acorn_index_path(table_name).is_file():
        return False
    metadata = read_kmeans_acorn_index_metadata(table_name)
    if not metadata:
        return False
    if str(metadata.get("table_name") or "") != table_name:
        return False
    if plan_id is not None:
        try:
            if int(metadata.get("plan_id")) != int(plan_id):
                return False
        except (TypeError, ValueError):
            return False
    if partition_id is not None and str(metadata.get("partition_id") or "") != str(partition_id):
        return False
    if vector_count is not None:
        try:
            if int(metadata.get("row_count")) != int(vector_count):
                return False
        except (TypeError, ValueError):
            return False
    if expected_index_kind is not None and str(metadata.get("index_kind") or "").lower() != str(expected_index_kind).lower():
        return False
    return True


def _run_helper(args: list[str]) -> dict[str, object]:
    binary = _binary_path()
    completed = subprocess.run(
        [str(binary), *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    payload_text = stdout.splitlines()[-1] if stdout else stderr.splitlines()[-1] if stderr else ""
    try:
        payload = json.loads(payload_text) if payload_text else {}
    except json.JSONDecodeError:
        payload = {}
    if completed.returncode != 0:
        detail = payload.get("error") if isinstance(payload, dict) else None
        if not detail:
            detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"KMeans ACORN helper failed: {detail}")
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"KMeans ACORN helper failed: {payload.get('error')}")
    return payload if isinstance(payload, dict) else {}


def build_kmeans_acorn_indexes(
    *,
    table_name: Optional[str] = None,
    force: bool = True,
    dimension: Optional[int] = None,
    hnsw_m: int = 32,
    acorn_m: int = 32,
    gamma: int = 12,
    m_beta: int = 64,
) -> None:
    args = ["build"]
    if table_name:
        args.extend(["--table", str(table_name)])
    if force:
        args.append("--force")
    if dimension is not None:
        args.extend(["--dimension", str(int(dimension))])
    args.extend(["--hnsw-m", str(int(hnsw_m))])
    args.extend(["--acorn-m", str(int(acorn_m))])
    args.extend(["--gamma", str(int(gamma))])
    args.extend(["--m-beta", str(int(m_beta))])
    _run_helper(args)


def drop_kmeans_acorn_indexes(table_names: Optional[Iterable[str]] = None) -> None:
    try:
        index_dir = kmeans_acorn_index_dir()
    except Exception:
        return
    if table_names is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return
    for table_name in table_names:
        for suffix in (".faiss", ".json"):
            path = index_dir / f"{table_name}{suffix}"
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _vector_to_text(query_vector) -> str:
    if isinstance(query_vector, str):
        return query_vector
    return "[" + ",".join(str(float(value)) for value in query_vector) + "]"


def search_kmeans_acorn(user_id: int, query_vector, topk: int, ef_search: int) -> tuple[list[tuple], float]:
    started_at = time.perf_counter()
    payload = _run_helper(
        [
            "search",
            "--user-id",
            str(int(user_id)),
            "--vector",
            _vector_to_text(query_vector),
            "--topk",
            str(int(topk)),
            "--ef-search",
            str(int(ef_search)),
        ]
    )
    results = []
    for row in payload.get("results", []) or []:
        if len(row) < 4:
            continue
        results.append((int(row[0]), int(row[1]), row[2], float(row[3])))
    helper_time = payload.get("time_seconds")
    if helper_time is None:
        helper_time = time.perf_counter() - started_at
    return results, float(helper_time)
