# Build Instructions

## Prerequisites

1. **Clone Faiss into this directory**

```bash
cd pointer_benchmark
git clone https://github.com/facebookresearch/faiss.git
```

You can now edit Faiss sources under `pointer_benchmark/faiss` and they will be rebuilt automatically.

2. **Install toolchain dependencies**

- CMake ≥ 3.24
- A C++17 compiler (Clang on macOS works fine)
- `libomp` (Homebrew: `brew install libomp`)
- BLAS/LAPACK (Apple's Accelerate framework is used by default on macOS)

3. **Verify database setup**

Ensure you have:
- PostgreSQL running
- Database configured in `../config.json`
- Tables: `documentblocks`, `documentblocks_role_*`, `RolePartitions`, `UserRoles`
- Query dataset at `../basic_benchmark/query_dataset.json`

## Building pointer_benchmark

```bash
cd pointer_benchmark
mkdir -p build && cd build
cmake ..
make
```

## Running

```bash
./pointer_benchmark
```

## Expected Output

```
Loaded 1000 queries
=== Building Shared Vector Table ===
Loaded 50000 vectors into shared vector table
Found 5 role partitions
=== Building Pointer HNSW Indices ===
Built pointer HNSW index for documentblocks_role_1 with 10000 vectors (shared storage)
...
=== Benchmarking Pointer HNSW ===
Results:
  Avg time: 12.34 ms
  P50: 10.5 ms
  P95: 25.6 ms
  P99: 35.2 ms
  Avg recall: 0.95

Results saved to /Users/.../pointer_benchmark/results/pointer_hnsw_results.json
```

## Troubleshooting

### "OpenMP not found" / "-fopenmp" errors
Ensure `libomp` is installed (`brew install libomp`) and re-run `cmake` after clearing your build directory. The `build_and_run.sh` script exports `OpenMP_ROOT` automatically when the Homebrew prefix is detected.

### "No such file or directory: config.json"
Ensure you're running from the correct directory and `../config.json` exists.

### "Connection refused"
Check PostgreSQL is running and config.json has correct connection details.

### `find_package(BLAS)`/`find_package(LAPACK)` failures
Set `BLA_VENDOR=Apple` (already done in `build_and_run.sh`) or install an alternative BLAS such as OpenBLAS, then reconfigure.
