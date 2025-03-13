/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `vecRaw` related functions for the SSE implementation of VecMath.
 */
#ifndef VECRAW_H_SSE
#define VECRAW_H_SSE
#include "vec_math_sse.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif


/// @brief Adds two raw vectors.
pure_fn vecRaw add_vecRaw(const vecRaw a, const vecRaw b, const unsigned N) {
    const unsigned n = (N+3)/4;
    vecRaw ret = (vecRaw)_mm_malloc(sizeof(__m128) * n, 64);
    #pragma unroll(4)
    for(unsigned i = 0; i < n; i++) {
        ret[i] = _mm_add_ps(a[i], b[i]);
    }
    return ret;
}

// /// @brief Subtracts two 4-vectors.
pure_fn vecRaw sub_vecRaw(const vecRaw a, const vecRaw b, const unsigned N) {
    const unsigned n = (N+3)/4;
    vecRaw ret = (vecRaw)_mm_malloc(sizeof(__m128) * n, 64);
    #pragma unroll(4)
    for(unsigned i = 0; i < n; i++) {
        ret[i] = _mm_add_ps(a[i], b[i]);
    }
    return ret;
}

// /// @brief Multiplies a 4-vector by a float.
// pure_fn vec4 scalar_mul_vec4(const vec4 a, const float b) {}

// /// @brief Multiplies two 4-vectors element-wise.
// pure_fn vec4 mul_vec4(const vec4 a, const vec4 b) {}

// /// @brief Divides a 4-vector by a float.
// pure_fn vec4 scalar_div_vec4(const vec4 a, const float b) {}

// /// @brief Divides two 4-vectors element-wise.
// pure_fn vec4 div_vec4(const vec4 a, const vec4 b) {}

// /// @brief Computes the dot product of two 4-vectors.
// pure_fn float dot_vec4(const vec4 a, const vec4 b) {}

// /// @brief Normalizes a 4-vector.
// pure_fn vec4 norm_vec4(const vec4 a) {}

// /// @brief Computes the length of a 4-vector.
// pure_fn float len_vec4(const vec4 a) {}

// /// @brief Computes the outer product of two 4-vectors as a `mat`
// pure_fn rmat4 outer_rvec4(const vec4 a, const vec4 b) {}



/// @brief Prints a vecRaw.
void print_vecRaw(const vecRaw a, const unsigned int N) {
    int i = 0;
    for(i = 0; i < N/4; i++) {
        printf("%f %f %f %f ",
            _mm_extractf_ps(a[i], 0),
            _mm_extractf_ps(a[i], 1),
            _mm_extractf_ps(a[i], 2),
            _mm_extractf_ps(a[i], 3));
    }
    switch(N & 3) { // last two bytes
        case 3:
            printf("%f %f %f", _mm_extractf_ps(a[i], 0), _mm_extractf_ps(a[i], 1), _mm_extractf_ps(a[i], 2));
            break;
        case 2:
            printf("%f %f", _mm_extractf_ps(a[i], 0), _mm_extractf_ps(a[i], 1));
            break;
        case 1:
            printf("%f", _mm_extractf_ps(a[i], 0));
            break;
    }
    printf("\n");
}





#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
