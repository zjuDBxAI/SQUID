# Latent Access Partitioning

## Goal

This module implements the system scaffold for a new multi-tenant retrieval idea:

- do **not** assume an explicit RBAC structure is given ahead of time
- instead, compress document-to-tenant visibility patterns into a small latent access space
- combine that latent access structure with semantic locality
- materialize partitions in the form of `semantic cell x latent atom`
- keep a residual path for documents whose access pattern is hard to compress
- route queries using both semantic relevance and tenant-to-atom compatibility
- apply exact permission filtering at query time so correctness is preserved

The intended long-term algorithm is:

1. Build a binary access matrix `A[document, tenant]`
2. Learn a compact latent access basis such that `A ~= ZB`
3. Use `Z` to assign each document a dominant latent atom
4. Partition documents by `(semantic_cell, latent_atom)`
5. Send high-residual documents to residual partitions
6. Route a query using the tenant's latent access profile and the query vector's semantic locality

## Important Status Note

The current codebase now contains a **working graph-regularized alternating trainer**. It is still not the final research version, but it is no longer just a clustering proxy or a thin heuristic stub.

What is implemented now:

- document-level access extraction from the existing RBAC schema
- a sparse alternating factorization trainer for `A ~= ZB`
- projected-gradient updates for document codes `Z` with simplex-style sparse projection
- semantic-group smoothing and local semantic kNN regularization during `Z` updates
- dead-atom reseeding for weak latent atoms
- richer training diagnostics and objective tracking
- residual estimation
- support-aware semantic-cell x latent-atom planning with cell-local atom budgets
- plan persistence and partition materialization
- block-level semantic enrichment for each partition through `semantic_anchor_vectors`
- ANN index creation for materialized partitions
- a benchmark-facing search path with exact permission filtering
- adaptive route expansion and a single-query `UNION ALL` routed execution path

What is intentionally left for the next iteration:

- improving online route scoring and fallback logic
- adding incremental update / retraining triggers
- exploring warm-start and partial-refit strategies for large access updates
- upgrading from heuristic atom reseeding into more principled split / merge / prune operators

This split is deliberate: the surrounding system interfaces are now stable enough that the trainer can be swapped without rewriting materialization, search, or benchmark tooling.

## Recent Improvements (2026-04-11)

The latest `latent_access` iteration improved the system in three concrete ways:

- each partition is now enriched with block-level semantic anchors (`semantic_anchor_vectors`) computed from the real block vectors inside that partition
- online routing now uses those anchors and can expand the route depth adaptively when semantic-cell competition is close
- routed search is now executed as a single `UNION ALL` SQL query instead of issuing one query per partition

These changes matter because the benchmark is block-level rather than document-level. A partition that looks coherent at the document-representative level may still miss the true top-k blocks unless routing sees richer block semantics.

A small smoke benchmark run on 2026-04-11 with:

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py benchmark \
  --enable-index true \
  --index-type hnsw \
  --statistics-type sql \
  --query-num 5 \
  --iterations 1 \
  --use-ground-truth-cache false
```

produced:

- `Average Recall: 0.5400`
- `Average Query Time: 0.0306 seconds`
- `Space used: 834.20 MB`

This is still below `dynamic_partition`, but it is a clear improvement over the earlier low-recall prototype and establishes a more credible base for the next optimization round.

## Conceptual Model

### Inputs

For each document `d`:

- a semantic vector `x_d`
- a visible tenant set `S_d`

The ideal mathematical model behind this module is:

- `A` is the binary document-tenant visibility matrix
- `Z` is the document-to-latent-atom assignment matrix
- `B` is the latent-atom-to-tenant compatibility matrix
- `A ~= ZB`

In the final version:

- `Z` should be sparse
- each document should activate one or a few atoms
- `B` should capture shared access structure across tenants
- large reconstruction error should trigger residual handling

### Physical Partition Shape

The physical partition key is:

- `semantic_cell_id`
- `latent_atom_id`
- `residual_flag`

So the main partition family is:

- `cell_<c>_atom_<a>`

And the exception family is:

- `cell_<c>_residual`

This makes the system more expressive than:

- one shared global index
- one index per tenant
- one index per explicit role combination

## Current File Layout

### `repository.py`

Responsibilities:

- extract document-level access examples from PostgreSQL
- fetch a representative vector per document
- fetch the tenant set associated with each document
- fetch document block counts for later physical size estimation
- fetch accessible document ids for exact filtering support

Current design choices:

- operates at **document level** for access compression
- still uses one representative block vector per document for training
- additionally fetches real block vectors per partition during materialization for routing-time semantic enrichment
- keeps the extraction logic independent from training logic

This is the correct layer to upgrade later if you want:

- mean pooling over blocks instead of first-block vectors
- richer document embeddings
- temporal access snapshots

### `trainer.py`

Responsibilities:

- define training configuration
- produce a trained latent access model object
- estimate document-level residuals
- expose tenant-to-atom scores

Current implementation:

- `PrototypeLatentAccessTrainer`
- builds the exact document-tenant access matrix `A`
- initializes sparse document codes from hybrid semantic/access clustering
- updates `B` with a ridge-regularized closed-form solve
- updates sparse document codes `Z` with projected gradient steps
- combines semantic group prototypes with local semantic kNN targets during `Z` updates
- periodically reseeds weak atoms from high-residual documents to avoid atom collapse
- keeps the outer loop stable by rejecting obviously worse factor updates
- exposes planner-facing stability controls such as `max_atoms_per_semantic_cell` and `min_partition_documents`
- uses `|A - ZB|` reconstruction error as the residual signal

The class name is still kept for compatibility, but the implementation is now a graph-regularized alternating optimizer.

A future version of this trainer should move further toward:

`reconstruction_loss + semantic_graph_regularization + structured_atom_lifecycle + update-aware_adaptation`

### `planner.py`

Responsibilities:

- take the trained model and document records
- cluster documents into semantic cells
- assign each document to a latent atom or residual path
- group documents into physical partitions
- attach metadata for routing and diagnostics

Outputs:

- `LatentAccessPartition`
- `LatentAccessPlan`

Current partition metadata includes:

- semantic centroid
- average residual
- max residual
- route prior

Current planning logic is not a raw `(cell, argmax_atom)` dump anymore. It first measures which atoms are stably supported inside each semantic cell, keeps only a small per-cell atom budget, and redirects weak local atom fragments either to a stronger stable atom in the same cell or to the cell residual path.

This layer is intentionally independent from how the trainer learns the atoms, but it now explicitly prevents over-fragmented physical plans.

### `load_result_to_database.py`

Responsibilities:

- define control-plane schema for the latest latent access plan
- persist partition metadata and atom-to-tenant weights
- enrich each partition with block-level semantic anchors before persistence
- materialize current partitions into concrete PostgreSQL tables
- create/drop ANN indexes for materialized tables
- expose one-shot helpers to build and materialize a plan

Control-plane tables introduced by this module:

- `latent_access_current_plan`
- `latent_access_current_partitions`
- `latent_access_current_partition_documents`
- `latent_access_current_atom_tenants`

Physical partition tables use prefix:

- `latent_documentblocks_partition_`

### `search.py`

Responsibilities:

- load the current latent access plan
- compute a tenant-specific route over current partitions
- execute ANN search over routed partitions only
- apply exact permission filtering inside SQL
- merge partition-local results into final top-k results

Current route score is still heuristic, but materially stronger than the initial scaffold:

- semantic similarity is computed against partition-level block anchors rather than a single coarse centroid
- tenant compatibility with the latent atom is still part of the score
- route prior from partition size is still used
- residual partitions still act as recall-preserving fallbacks
- route depth now expands adaptively when semantic-cell scores are too close
- routed SQL execution is now issued as one `UNION ALL` query instead of one query per partition

This is enough for a benchmarkable prototype and is already meaningfully better than the earlier low-recall version.

A later version should incorporate:

- better learned route scoring
- calibrated route confidence
- per-tenant fallback for low-confidence routes

### `__init__.py`

Exports the public interface for the latent access module.

## Benchmark Entry Points

The following wrappers are added under `basic_benchmark/`:

- `build_latent_access_partitions.py`
- `build_latent_access_indexes.py`
- `initialize_latent_access_tables.py`
- `test_latent_access.py`

They mirror the style already used by `dynamic_partition` and `adaptive_tenant`.

The benchmark registry is wired through:

- `basic_benchmark/condition_config.py`
- `basic_benchmark/space_calculate.py`


## Experiment CLI

A unified experiment CLI is available at:

- `controller/latent_access/cli.py`

Recommended way to run it:

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py --help
```

Available subcommands:

- `init`: initialize control-plane tables
- `build`: train, plan, and materialize a latent access layout
- `summary`: inspect the current materialized plan
- `index`: build ANN indexes for the current partitions
- `benchmark`: run the benchmark-facing search experiment
- `clear`: drop current materialized partitions and clear current plan metadata

A recommended experiment flow is:

1. Initialize the control-plane schema

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py init
```

2. Build a latent access layout with a compact starting configuration

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py build \
  --atom-count 16 \
  --semantic-cell-count 24 \
  --residual-quantile 0.95 \
  --semantic-knn 8 \
  --semantic-knn-weight 0.15 \
  --max-atoms-per-semantic-cell 3 \
  --min-partition-documents 8 \
  --sparsity 2 \
  --max-iterations 10 \
  --training-limit 2000
```

3. Inspect the resulting plan statistics

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py summary
```

4. Build ANN indexes after the partition count looks reasonable

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py index --index-type hnsw
```

5. Run the benchmark experiment

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py benchmark \
  --enable-index true \
  --index-type hnsw \
  --statistics-type sql \
  --query-num 1000 \
  --iterations 1
```

6. Clear current experiment state before another large rebuild if needed

```bash
cd /data/Multitenanthakes
/home/chenyang/.conda/envs/multitenant/bin/python controller/latent_access/cli.py clear
```

Practical tuning guidance:

- If partition count is too high, first reduce `semantic-cell-count` or `atom-count`, then tighten `max-atoms-per-semantic-cell` or increase `min-partition-documents`.
- If too many documents are redirected into residual partitions, slightly relax `min-partition-documents` or increase `max-atoms-per-semantic-cell`.
- If training is too slow, reduce `training-limit`, `semantic-knn`, or `max-iterations` first.
- If recall drops too much, first inspect route coverage, then increase `route_limit` or rerun with a less aggressive partition budget.
- The current default benchmark path is intentionally stronger than the earliest prototype: `route_limit=16`, `partition_fetch_multiplier=6`, and adaptive route expansion stays enabled when `route_limit` is not passed explicitly.

## End-to-End Flow

### Offline Build

1. Read document-level access records from the current database
2. Train the prototype latent access model
3. Build semantic cells
4. Assign each document to `(cell, atom)` or residual
5. Persist the plan
6. Materialize partition tables
7. Optionally build ANN indexes

### Online Query

1. Receive `(tenant_id, query_vector)`
2. Load the latest plan and atom-to-tenant weights
3. Score candidate partitions using tenant compatibility and semantic proximity
4. Search the top routed partitions only
5. Enforce exact permission checks inside SQL
6. Merge results across partitions

## Why Exact Permission Filtering Still Exists

The latent access model is used for:

- partition construction
- routing
- reducing search fanout

It is **not** used as the final authorization mechanism.

This is important.

Even if the latent model is imperfect:

- it may cause suboptimal routing
- it may increase residual traffic
- it may hurt recall if the route is too narrow

But it should **not** create authorization leaks, because the actual SQL query still checks whether the tenant can access each document.

## Current Limitations

1. The current factorization trainer uses simple alternating updates rather than a more principled optimizer.
2. Semantic regularization is group-based, not yet graph-based.
3. The representative vector is currently one block vector per document.
4. Route scoring is heuristic and does not yet use a calibrated cost model.
5. There is no incremental retraining path yet.
6. Residual handling is static and threshold-based.

These are expected first-version constraints.

## Recommended Next Steps

### 1. Strengthen the factorization trainer

Target file:

- `controller/latent_access/trainer.py`

Suggested upgrade:

- replace the current heuristic alternating updates with a stronger optimizer
- add graph-based semantic regularization
- add better atom birth / merge control
- keep sparse document assignments

### 2. Improve document-level semantic representation

Target file:

- `controller/latent_access/repository.py`

Suggested upgrade:

- use mean-pooled block vectors or another document embedding
- optionally add text-side signals if available

### 3. Improve route scoring

Target file:

- `controller/latent_access/search.py`

Suggested upgrade:

- use learned route scores
- support route expansion when confidence is low
- integrate a better latency-aware cost term

### 4. Add incremental update support

Target files:

- `controller/latent_access/trainer.py`
- `controller/latent_access/load_result_to_database.py`

Suggested upgrade:

- freeze `B` for online projection
- incrementally encode new documents
- trigger local rebuilds when residuals drift upward
- periodically retrain the full basis offline

## Intended Research Positioning

This module is meant to support a method with the following positioning:

- unlike Honeybee-style systems, it does not assume explicit predicates or RBAC roles are already known
- unlike role mining, it does not aim to recover human-interpretable roles
- unlike pure shared-index multi-tenant systems, it explicitly injects compressed access structure into physical routing and partitioning

The core claim is:

- multi-tenant retrieval can be improved by learning a compact latent access space directly from document visibility patterns, then coupling that latent access space with semantic locality during partitioning and routing

## Quick Usage

Build partitions:

```bash
python basic_benchmark/build_latent_access_partitions.py --atom-count 32 --semantic-cell-count 64
```

Build indexes:

```bash
python basic_benchmark/build_latent_access_indexes.py --index-type hnsw
```

Run the benchmark:

```bash
python basic_benchmark/test_latent_access.py --enable-index true --index-type hnsw
```

## Summary

The code in this directory is the **system skeleton** for latent-access-driven multi-tenant retrieval.

What is stable already:

- module boundaries
- persistence schema
- materialization flow
- benchmark integration
- search interface

What should change next:

- the quality of the training algorithm inside `trainer.py`
- the semantic regularization and online update strategy around it

That is exactly where the research novelty should now be pushed.
