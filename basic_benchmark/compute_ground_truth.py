"""Utility to compute ground-truth results for a query dataset.

Loads queries from JSON, uses the legacy ground-truth helpers in
``basic_benchmark.common_function`` to execute the expensive lookup, and writes
pointer-friendly output alongside the existing cache file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from basic_benchmark.common_function import (  # type: ignore  # noqa: E402
    _save_ground_truth_cache,
    ground_truth_func_batch,
    set_ground_truth_total_queries,
)


def load_queries(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of queries in {path}")
    return data


def to_pointer_results(results: Iterable[Iterable]) -> List[List[List[int]]]:
    pointer_results: List[List[List[int]]] = []
    for query_results in results:
        formatted: List[List[int]] = []
        for entry in query_results or []:
            if not entry or len(entry) < 2:
                continue
            block_id, document_id = entry[0], entry[1]
            try:
                block_id = int(block_id)
            except (TypeError, ValueError):
                continue
            try:
                document_id = int(document_id)
            except (TypeError, ValueError):
                continue
            formatted.append([block_id, document_id])
        pointer_results.append(formatted)
    return pointer_results


def write_pointer_cache(path: Path, pointer_results: List[List[List[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(pointer_results, fh, indent=2)


def load_pointer_cache(path: Path) -> Optional[List[List[List[int]]]]:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except json.JSONDecodeError:
        return None
    return None


def load_basic_cache_if_valid(queries: List[dict]) -> Optional[List]:
    cache_path = PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json"
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
    except json.JSONDecodeError:
        return None

    if not isinstance(cached, list) or len(cached) != len(queries):
        return None

    # Pointer-style cache (list of lists) – cannot reuse as-is
    if cached and isinstance(cached[0], list):
        return None

    def query_matches(expected: dict, actual: dict) -> bool:
        return (
            expected.get("user_id") == actual.get("user_id") and
            expected.get("topk", 5) == actual.get("topk", 5) and
            expected.get("query_vector") == actual.get("query_vector")
        )

    if queries and cached:
        try:
            first_matches = query_matches(queries[0], cached[0].get("query", {}))
            last_matches = query_matches(queries[-1], cached[-1].get("query", {}))
        except AttributeError:
            return None
    else:
        first_matches = True
        last_matches = True
    if not (first_matches and last_matches):
        return None

    return [item.get("ground_truth", []) for item in cached]


def rewrite_basic_cache_from_pointer(pointer_results: List[List[List[int]]], queries: List[dict]) -> None:
    cache_path = PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json"
    reconstructed = []
    for idx, query_results in enumerate(pointer_results):
        formatted = []
        for entry in query_results:
            if not entry or len(entry) < 2:
                continue
            block_id, document_id = entry[0], entry[1]
            try:
                block_id = int(block_id)
                document_id = int(document_id)
            except (TypeError, ValueError):
                continue
            formatted.append((block_id, document_id, "", 0.0))
        reconstructed.append(formatted)

    _save_ground_truth_cache(queries, reconstructed, str(cache_path))
    print(f"✓ Reconstructed basic_benchmark cache from pointer results at {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ground-truth results for a query dataset")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=PROJECT_ROOT / "basic_benchmark" / "query_dataset.json",
        help="Path to the query dataset JSON",
    )
    parser.add_argument(
        "--pointer-output",
        type=Path,
        default=PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json",
        help="Where to write pointer-style ground-truth results",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Force FAISS ground-truth generation (falls back to Postgres if unavailable)",
    )
    parser.add_argument(
        "--no-basic-cache",
        action="store_true",
        help="Skip updating basic_benchmark/ground_truth_cache.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if caches already exist",
    )

    args = parser.parse_args()

    if not args.query_file.exists():
        raise FileNotFoundError(f"Query dataset not found: {args.query_file}")

    queries = load_queries(args.query_file)
    pointer_results: Optional[List[List[List[int]]]] = None
    results: Optional[List] = None

    if not args.force:
        pointer_cache = load_pointer_cache(args.pointer_output)
        if pointer_cache is not None and len(pointer_cache) == len(queries):
            pointer_results = pointer_cache

        results = load_basic_cache_if_valid(queries)

        if results is not None and pointer_results is not None:
            print("✓ Ground truth caches already present; nothing to do")
            return

        if results is not None and pointer_results is None:
            pointer_results = to_pointer_results(results)
            write_pointer_cache(args.pointer_output, pointer_results)
            print(f"✓ Wrote pointer ground truth for {len(pointer_results)} queries to {args.pointer_output}")
            return

        if results is None and pointer_results is not None:
            rewrite_basic_cache_from_pointer(pointer_results, queries)
            print("✓ Reused pointer cache to refresh basic ground truth; nothing else to do")
            return

    set_ground_truth_total_queries(len(queries))
    results = ground_truth_func_batch(
        queries,
        use_faiss=True if args.use_faiss else None,
        use_cache=not args.no_basic_cache,
    )

    pointer_results = to_pointer_results(results)
    write_pointer_cache(args.pointer_output, pointer_results)

    print(f"✓ Wrote pointer ground truth for {len(pointer_results)} queries to {args.pointer_output}")


if __name__ == "__main__":
    main()
