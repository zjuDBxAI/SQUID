# Pointer Benchmark Usage Guide

## Build Instructions

```bash
cd pointer_benchmark
mkdir build && cd build
cmake ..
make
```

## Prerequisites

1. Database must be set up with:
   - `documentblocks` table with vectors
   - Role partition tables (`documentblocks_role_*`)
   - `RolePartitions`, `UserRoles` tables
   - Query dataset at `../basic_benchmark/query_dataset.json`

2. Config file at project root: `../config.json`

## Running the Benchmark

```bash
./pointer_benchmark
```

## What It Does

1. **Loads shared vector table**: All vectors from `documentblocks` into RAM
2. **Builds pointer HNSW indices**: One HNSW graph per role partition, all sharing the same vector table
3. **Runs queries**: Tests query performance with pointer-based access
4. **Computes metrics**: Latency (avg, P50/P95/P99), recall@k

## Output

Results saved to `results/pointer_hnsw_results.json`:

```json
{
  "config": {
    "M": 32,
    "efConstruction": 200,
    "efSearch": 100,
    "topk": 10
  },
  "results": {
    "avg_time_ms": 12.34,
    "p50_ms": 10.5,
    "p95_ms": 25.6,
    "p99_ms": 35.2,
    "avg_recall": 0.95,
    "num_queries": 1000,
    "num_partitions": 5,
    "shared_table_size": 100000
  }
}
```

## Comparing with Standard HNSW

To compare with standard HNSW (from `acorn_benchmark`), run both benchmarks and compare:
- Query latency (pointer version will be slower due to extra indirection)
- Memory usage (pointer version uses less total memory for multiple partitions)
- Recall should be identical

## Tuning Parameters

Edit `src/main.cpp` to adjust:
- `M`: HNSW graph connectivity (16-64)
- `efConstruction`: Build-time search depth (100-400)
- `efSearch`: Query-time search depth (10-200)
- `topk`: Number of results to return