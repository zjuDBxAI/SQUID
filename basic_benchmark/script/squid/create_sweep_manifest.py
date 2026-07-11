#!/usr/bin/env python3
"""Create a database-free build manifest for a versioned memory sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from plan_manifest import METHODS, create_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create versioned materialization manifests without PostgreSQL access")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--memory-ratios", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--version-prefix", default="sweep")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for method in args.methods:
        for ratio in args.memory_ratios:
            version = f"{args.version_prefix}_{str(ratio).replace('.', 'p')}"
            manifest = create_manifest(method, ratio, version).as_dict()
            manifest["state"] = "planned"
            manifest["build_contract"] = {
                "metadata_relations": [
                    manifest["plan_relation"],
                    manifest["partition_relation"],
                    manifest["route_relation"],
                    manifest["pattern_relation"],
                ],
                "partition_prefix": manifest["table_prefix"],
                "query_selector": {"method": method, "memory_ratio": ratio},
            }
            entries.append(manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {len(entries)} planned materializations to {args.output}")


if __name__ == "__main__":
    main()
