cmake_minimum_required(VERSION 3.20)
project(pointer_benchmark)

set(CMAKE_CXX_STANDARD 17)

# Enable maximum compiler optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")

# Add include directories
include_directories(src)
include_directories(/usr/local/include)
include_directories(/usr/local/pgsql/include)
include_directories(/opt/homebrew/include)
include_directories(/opt/homebrew/opt/libomp/include)

# Link directories
link_directories(/usr/local/pgsql/lib)
link_directories(/opt/homebrew/lib)

# Find required libraries
find_library(PQXX_LIB pqxx REQUIRED PATHS /opt/homebrew/lib /usr/local/lib)
find_library(PQ_LIB pq REQUIRED PATHS /usr/local/pgsql/lib /opt/homebrew/lib)

# Find nlohmann-json library
find_package(nlohmann_json REQUIRED)

# ---------------------------------------------------------------------------
# Build an in-tree Faiss (CPU-only, no extras) so we can modify the library.
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_CUVS OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_ROCM OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_MKL OFF CACHE BOOL "" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(_libomp_prefix "")
if(APPLE)
    # Help Faiss' CMake find libomp and prefer Accelerate for BLAS/LAPACK.
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(_libomp_prefix "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(_libomp_prefix "/usr/local/opt/libomp")
    endif()
    if(_libomp_prefix)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_libomp_prefix}/include" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${_libomp_prefix}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
        list(APPEND CMAKE_PREFIX_PATH "${_libomp_prefix}")
    else()
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
    endif()
    set(BLA_VENDOR "Apple" CACHE STRING "" FORCE)
endif()

add_subdirectory(faiss)

# Add OpenMP flags for parallelization
find_package(OpenMP)
if(APPLE)
    if(_libomp_prefix)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${_libomp_prefix}/include")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${_libomp_prefix}/lib -lomp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
    endif()
elseif(OpenMP_CXX_FOUND)
    # Linux: use system OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Source files for role partition comparison (main benchmark)
set(SOURCES_ROLE_COMPARISON
    src/main_role_partition_comparison.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/independent_hnsw_index.cpp
    src/global_hnsw_index.cpp
)

# Build the main role partition comparison executable
add_executable(role_partition_comparison ${SOURCES_ROLE_COMPARISON})

# Global postfilter comparison (logical vs physical)
add_executable(global_postfilter_comparison
    src/main_global_postfilter_comparison.cpp
    src/shared_vector_table.cpp
    src/global_hnsw_index.cpp
)

# Preload variant that keeps independent partitions resident in a contiguous arena.
add_executable(pointer_benchmark_preload
    src/main_preload.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/independent_hnsw_index.cpp
    src/global_hnsw_index.cpp
)

# Pointer-only executable (no independent scenario)
add_executable(pointer_benchmark_pointeronly
    src/main_pointer_only.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/independent_hnsw_index.cpp
    src/global_hnsw_index.cpp
)

add_executable(pointer_benchmark_pointerpreload
    src/main_pointer_preload.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/independent_hnsw_index.cpp
    src/global_hnsw_index.cpp
)

# Link libraries for main executables
target_link_libraries(role_partition_comparison PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

target_link_libraries(global_postfilter_comparison PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

target_link_libraries(pointer_benchmark_preload PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

target_link_libraries(pointer_benchmark_pointeronly PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

target_link_libraries(pointer_benchmark_pointerpreload PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

# Platform-specific optimizations
if(APPLE)
    target_link_libraries(role_partition_comparison PRIVATE "-framework Accelerate")
    target_link_libraries(global_postfilter_comparison PRIVATE "-framework Accelerate")
    target_link_libraries(pointer_benchmark_preload PRIVATE "-framework Accelerate")
    target_link_libraries(pointer_benchmark_pointeronly PRIVATE "-framework Accelerate")
    target_link_libraries(pointer_benchmark_pointerpreload PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(role_partition_comparison PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(global_postfilter_comparison PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(pointer_benchmark_preload PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(pointer_benchmark_pointeronly PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(pointer_benchmark_pointerpreload PRIVATE OpenMP::OpenMP_CXX)
endif()

# Test executables
# Test 1: Role Partition
add_executable(test_logical_role_partition
    src/tests/test_logical_partition/test_role_partition.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
)

target_link_libraries(test_logical_role_partition PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_logical_role_partition PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_logical_role_partition PRIVATE OpenMP::OpenMP_CXX)
endif()

# Test 1: Physical Role Partition
add_executable(test_physical_role_partition
    src/tests/test_physical_partition/test_role_partition.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/independent_hnsw_index.cpp
)

target_link_libraries(test_physical_role_partition PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_physical_role_partition PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_physical_role_partition PRIVATE OpenMP::OpenMP_CXX)
endif()


# Test 2: Postfilter
add_executable(test_logical_postfilter
    src/tests/test_logical_partition/test_postfilter.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
    src/global_hnsw_index.cpp
)

target_link_libraries(test_logical_postfilter PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_logical_postfilter PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_logical_postfilter PRIVATE OpenMP::OpenMP_CXX)
endif()


# Test 2: Physical Postfilter
add_executable(test_physical_postfilter
    src/tests/test_physical_partition/test_postfilter.cpp
    src/shared_vector_table.cpp
    src/global_hnsw_index.cpp
)

target_link_libraries(test_physical_postfilter PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_physical_postfilter PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_physical_postfilter PRIVATE OpenMP::OpenMP_CXX)
endif()


# Test 3: Logical Dynamic Partition
add_executable(test_logical_dynamic_partition
    src/tests/test_logical_partition/test_dynamic_partition.cpp
    src/shared_vector_table.cpp
    src/pointer_hnsw_index.cpp
)

target_link_libraries(test_logical_dynamic_partition PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_logical_dynamic_partition PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_logical_dynamic_partition PRIVATE OpenMP::OpenMP_CXX)
endif()

# Test 4: Physical Dynamic Partition
add_executable(test_physical_dynamic_partition
    src/tests/test_physical_partition/test_dynamic_partition.cpp
    src/shared_vector_table.cpp
    src/independent_hnsw_index.cpp
)

target_link_libraries(test_physical_dynamic_partition PRIVATE
    ${PQXX_LIB}
    ${PQ_LIB}
    faiss
    nlohmann_json::nlohmann_json
)

if(APPLE)
    target_link_libraries(test_physical_dynamic_partition PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_physical_dynamic_partition PRIVATE OpenMP::OpenMP_CXX)
endif()



# Test 5: HNSW implementation comparison
add_executable(test_hnsw_compare
    src/tests/test_hnsw_compare.cpp
)

target_link_libraries(test_hnsw_compare PRIVATE
    faiss
)

if(APPLE)
    target_link_libraries(test_hnsw_compare PRIVATE "-framework Accelerate")
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(test_hnsw_compare PRIVATE OpenMP::OpenMP_CXX)
endif()
