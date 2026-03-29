# HQI QD-tree Utilities

This directory contains the scripts used to build, persist, inspect, and
visualise the HQI role-aware QD-tree. The most common commands are listed
below for quick reference. Replace paths and parameters as needed.

## Build the QD-tree

```bash
# Example: build using the default workload-aware settings
python controller/baseline/HQI/build_tree.py --min-size 10000
```

Only `--min-size` is exposed; it controls the minimum number of blocks per leaf. The tree now grows
until the min-size condition stops further splits (no max-depth cap). All other builder parameters
use the repository defaults.

## Persist partitions to PostgreSQL

```bash
python controller/baseline/HQI/persist_tree.py --workers 8
```


## Debugging tools

Inspect a specific query and optionally run SQL against the candidate partitions:

```bash
python controller/baseline/HQI/debug/debug_qdtree_query.py \
  --query-json basic_benchmark/query_dataset.json \
  --query-index 0 \
  --run-sql
```

List the partitions reachable by the first N roles:

```bash
python controller/baseline/HQI/debug/list_role_partitions.py --limit 10
```

Validate that each materialised partition only contains documents visible to
its required roles:

```bash
python controller/baseline/HQI/debug/validate_qdtree_partitions.py
```

## Visualising the tree

Export a Graphviz DOT file (limit depth for readability):

```bash
python controller/baseline/HQI/debug/export_qdtree_dot.py \
  --max-depth 8 \
  --include-document-roles
```

Render with Graphviz:

```bash
dot -Tpdf qd_tree_depth8.dot -o qd_tree_depth8.pdf
```

## Run controller tests

Execute the pointer benchmark tests that exercise the HQI QD-tree pipeline:

```bash
cd basic_benchmark
python test_all.py --algorithm QDTree --efs 10
```

Adjust the `--efs` values to the search depths you want to benchmark.

## Additional notes

- Connection details (tree path, partition prefix, output location, database
  credentials) should be managed via `config.json` or environment variables,
  so the commands above only specify parameters that differ per run.
- Use the same partition prefix consistently across build, persist, and
  debugging commands.
