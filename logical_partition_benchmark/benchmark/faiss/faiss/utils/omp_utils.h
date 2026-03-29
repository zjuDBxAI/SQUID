#pragma once

#ifdef _OPENMP
#include <omp.h>

namespace faiss {

inline int omp_get_max_threads() {
    return ::omp_get_max_threads();
}

inline int omp_get_thread_num() {
    return ::omp_get_thread_num();
}

inline int omp_get_num_threads() {
    return ::omp_get_num_threads();
}

inline int omp_in_parallel() {
    return ::omp_in_parallel();
}

inline int omp_get_nested() {
    return ::omp_get_nested();
}

inline void omp_set_nested(int flag) {
    ::omp_set_nested(flag);
}

} // namespace faiss

#else

namespace faiss {

inline int omp_get_max_threads() {
    return 1;
}

inline int omp_get_thread_num() {
    return 0;
}

inline int omp_get_num_threads() {
    return 1;
}

inline int omp_in_parallel() {
    return 0;
}

inline int omp_get_nested() {
    return 0;
}

inline void omp_set_nested(int) {}

} // namespace faiss

#endif
