# Versioned Materialized Plans

This directory defines the storage contract for keeping multiple materialized
benchmark plans at once. A plan is selected by its method and storage
amplification (called `memory_ratio` in the CLI).

## Plan Identity

Each version has one immutable identity:

```text
(method, memory_ratio) -> registry_id -> plan_id -> namespaced tables
```

Supported methods are `ours`, `honeybee`, `veda`, and `effveda`.

Physical table names must be namespaced by the registry id. Examples:

```text
ours_m2p0_v101_private_0
honeybee_m2p0_v201_partition_17
veda_m2p0_v301_node_r_7_13
effveda_m2p0_v401_node_r_7_13
```

The current fixed prefixes (`kmeans_documentblocks_partition_*`,
`documentblocks_partition_*`, and `veda_documentblocks_node_*`) cannot be
used for persistent multi-ratio plans because a later build overwrites them.

## Registry

`schema.sql` creates two metadata tables:

- `benchmark_plan_registry`: one ready/building plan per method and ratio.
- `benchmark_plan_relations`: every physical table owned by that plan.

The registry is intentionally separate from existing `*_current_*` metadata.
It lets a query resolve a plan without scanning other ratio versions.

## CLI

The CLI does not build or delete a plan. It only initializes, registers, and
resolves metadata. This separation avoids touching an active database while
data is being loaded.

```bash
# Run only after the target database is ready.
venv/bin/python basic_benchmark/script/squid/versioned_plan_registry.py init

venv/bin/python basic_benchmark/script/squid/versioned_plan_registry.py register \
  --method ours --memory-ratio 2.0 --plan-id 101 \
  --table-prefix ours_v101 --state ready

venv/bin/python basic_benchmark/script/squid/versioned_plan_registry.py resolve \
  --method ours --memory-ratio 2.0
```

`plan-id` is the method-local metadata plan id after the corresponding builder
has been updated to preserve versions. The next implementation step is to make
each builder generate table names from the registry id and avoid current-plan
deletes/stale-table cleanup for versioned builds.

## Query Contract

A benchmark command receives both options:

```text
--method ours --memory-ratio 2.0
```

It resolves exactly one `registry_id`, loads only that plan's route metadata,
and issues ANN queries only against relations recorded for that registry id.
No query should inspect another memory ratio's tables.

## Deterministic Manifest

Generate names without connecting to PostgreSQL:

```bash
venv/bin/python basic_benchmark/script/squid/versioned_plan_registry.py manifest \
  --method ours --memory-ratio 2.0 --version 101
```

After a builder has created a version, record every partition table, index, and
namespaced route/pattern metadata relation with `register-relations`.

## Sweep Manifest

Create the six-ratio, four-method build plan without connecting to PostgreSQL:

```bash
venv/bin/python basic_benchmark/script/squid/create_sweep_manifest.py \
  --methods ours honeybee veda effveda \
  --memory-ratios 1.0 2.0 3.0 4.0 5.0 6.0 \
  --version-prefix yfcc_sweep \
  --output basic_benchmark/script/squid/yfcc_sweep_manifest.json
```

This only records the naming/build contract. It does not build partitions or
access PostgreSQL. A later version-aware builder consumes one manifest entry at
a time, registers it as `building`, materializes only its namespaced relations,
measures space, and finally marks it `ready`.
