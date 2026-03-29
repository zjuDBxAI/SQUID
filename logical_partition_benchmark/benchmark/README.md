# Pointer-based HNSW Benchmark

This directory contains the C++ benchmark suite that compares **logical pointer
partitions** (graphs share a single vector table) against **physical partitions**
where every role keeps a complete HNSW index. It is meant to be used alongside
the Python drivers under `basic_benchmark/`.

## 1. Prerequisites

Before building the C++ binaries, populate the database and generate queries via
the Python utilities in the project root:

```bash
cd basic_benchmark
python initialize_role_partition_tables.py --index_type hnsw
python initialize_combination_role_partition_tables.py --index_type hnsw  # optional
python generate_queries.py --num_queries 1000 --topk 10 --num_threads 4
python compute_ground_truth.py
```

For dynamic partitions, run the AnonySys driver once so that the materialized
tables exist:

```bash
cd controller/dynamic_partition/hnsw
python AnonySys_dynamic_partition.py --storage 1.2 --recall 0.95
```

The benchmark expects a `config.json` in this folder. At minimum it must define
`index_storage_path`, the directory where HNSW graphs are written (see
`config.json` for an example).

## 2. Build

```bash
cd logical_partition_benchmark/benchmark
mkdir -p build
cd build
cmake ..
make -j
```

Alternatively, `./build_and_run.sh` will compile the project in `build2/` and run
the default scenario afterwards.

## 3. Run Scenarios



## 4. Tests

The CMake project builds six end-to-end validation binaries (three logical,
three physical) plus a lightweight HNSW comparison utility:

| Test binary | Scenario |
|-------------|----------|
| `test_logical_role_partition` | Pointer HNSW on role partitions |
| `test_physical_role_partition` | Independent HNSW per role (physical baseline) |
| `test_logical_postfilter` | Pointer HNSW global/postfilter |
| `test_physical_postfilter` | Physical global/postfilter baseline |
| `test_logical_dynamic_partition` | Pointer HNSW for dynamic partitions |
| `test_physical_dynamic_partition` | Physical dynamic-partition baseline |
| `test_hnsw_compare` | Sanity check: pointer vs independent HNSW search parity |

They rely on the same database tables, query datasets, and `config.json`
described earlier.

Run the whole suite after building:

```bash
cd build
ctest --output-on-failure
```

Or invoke a specific scenario directly, e.g.:

```bash
./test_logical_dynamic_partition
./test_physical_dynamic_partition
```

Each executable accepts optional tuning flags (`--warmup` defaults to 0; `--ef-search`
defaults are the values shown below):

```bash
# Logical pointer-based benchmarks
./test_logical_role_partition      --ef-search 15
./test_logical_postfilter          --ef-search 600
./test_logical_dynamic_partition   --ef-search 30

# Physical baselines
./test_physical_role_partition     --ef-search 15
./test_physical_postfilter         --ef-search 600
./test_physical_dynamic_partition  --ef-search 30
```

Use `--help` on any binary for a quick reference.
