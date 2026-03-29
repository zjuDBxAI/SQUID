cmake_minimum_required(VERSION 3.20)
project(acorn_benchmark)

set(CMAKE_CXX_STANDARD 17)

# Add include directories
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/ACORN_lib) # Use faiss/impl as include directory
include_directories(/usr/local/include)
include_directories(/usr/local/pgsql/include)
include_directories(/opt/homebrew/include)
include_directories(/opt/homebrew/opt/libomp/include) # Include libomp headers
link_directories(/usr/local/pgsql/lib)
link_directories(/opt/homebrew/lib)
link_directories(${CMAKE_SOURCE_DIR}/ACORN_lib/build/faiss)

# Find required libraries
find_library(PQXX_LIB pqxx REQUIRED PATHS /opt/homebrew/lib /usr/local/lib)
find_library(PQ_LIB pq REQUIRED PATHS /usr/local/pgsql/lib /opt/homebrew/lib)

# Find nlohmann-json library
find_package(nlohmann_json REQUIRED)

# Add OpenMP support, falling back gracefully if unavailable
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Compiling with OpenMP support.")
else()
    message(WARNING "OpenMP not found; continuing without parallel CPU acceleration.")
endif()

# Set RPATH to include libomp and ACORN libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/ACORN_lib/build/faiss")
set(CMAKE_BUILD_RPATH "${CMAKE_SOURCE_DIR}/ACORN_lib/build/faiss")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Source files
file(GLOB SOURCES src/*.cpp)

# Locate FAISS shared library produced by ACORN build
find_library(FAISS_LIB
        NAMES faiss libfaiss
        PATHS
        ${CMAKE_SOURCE_DIR}/ACORN_lib/build/faiss
        ${CMAKE_SOURCE_DIR}/ACORN_lib/build/lib
        NO_DEFAULT_PATH
        REQUIRED)

# Build the executable
add_executable(main ${SOURCES}
        include/acorn_search.h
        src/acorn_search.cpp
        include/index_creation.h
        src/index_creation.cpp
        include/benchmark_utils.h
        src/benchmark_utils.cpp
        include/dynamic_partition_search.h
        src/dynamic_partition_search.cpp
        include/json_utils.h
        src/json_utils.cpp
        include/row_level_security.h
        src/row_level_security.cpp
)

# Link libraries, including the prebuilt ACORN library
target_link_libraries(main PRIVATE
        ${PQXX_LIB}
        ${PQ_LIB}
        nlohmann_json::nlohmann_json
        ${FAISS_LIB}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()
