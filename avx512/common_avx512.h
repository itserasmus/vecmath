/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Common file containing all the definitions for the AVX512 implementatio of VecMath.
 */

#ifndef VEC_MATH_COMMON_H_AVX512
#define VEC_MATH_COMMON_H_AVX512
#include "vec_math_avx512.h"

#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
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
/// @details The elements are stored as a `__m512`. The first three lanes
/// store the rows, with the first three values of each lane being the
/// elements of the matrix and the last row and last value of every lane
/// being undefined (not necessarily zero)
typedef __m512 mat3;
/// @brief A row major 4x4 float matrix
typedef __m512 rmat4;

/// @brief A 4x4 float matrix composed of 2x2 blocks. For a row
/// major matrix, use `rmat4` instead.
/// @details It is stored as
/// \code
/// a0 a1 | b0 b1
/// a2 a3 | b2 b3
/// -------------
/// c0 c1 | d0 d1
/// c2 c3 | d2 d3
/// \endcode
/// where a, b, c and d are the 4 lanes of a `__m512`
typedef __m512 mat4;


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
