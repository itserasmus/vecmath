/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Common file containing all the definitions for the SSE implementatio of VecMath.
 */

#ifndef VEC_MATH_COMMON_H_AVX
#define VEC_MATH_COMMON_H_AVX
#include "vec_math_avx.h"
#include <stdalign.h>

#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif


/// @brief A 2 element float vector
typedef __m128 vec2;
/// @brief A 3 element float vector
typedef __m128 vec3;
/// @brief A 4 element float vector
typedef __m128 vec4;

/// @brief A row major 2x2 float matrix
typedef __m128 mat2;
/// @brief A row major 3x3 float matrix. The AVX `mat3` does not offer
/// much of a performance gain over the SSE version.
/// @details The elements are stored as a `__m256` and a `__m128`. The
/// `__m256` contains the first two rows, and the `__m128` contains the
/// last row, with the first three values of each lane being the elements
/// of the matrix and the last value being undefined (not necessarily zero)
typedef struct mat3 {
    __m256 m0;
    __m128 m1;
} VECM_ALIGN_64 mat3;
/// @brief A row major 4x4 float matrix
typedef struct rmat4 {
    __m256 m0;
    __m256 m1;
} VECM_ALIGN_64 rmat4;

/// @brief A 4x4 float matrix composed of 2x2 blocks. For a row
/// major matrix, use `rmat4` instead.
/// @details It is stored as
/// \code
/// a0 a1 | b0 b1
/// a2 a3 | b2 b3
/// -------------
/// b4 b5 | a4 a5
/// b6 b7 | a6 a7
/// \endcode
/// where a and d are the 2 `__m256`s
typedef struct mat4 {
    __m256 b0;
    __m256 b1;
} VECM_ALIGN_64 mat4;


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
