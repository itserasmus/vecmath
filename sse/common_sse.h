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

#ifndef VEC_MATH_COMMON_H_SSE
#define VEC_MATH_COMMON_H_SSE
#include "vec_math_sse.h"

#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#define __attribute__(x)
#endif
#if !defined(__MSC_VER) && !defined(__clang__)
#define __declspec(x)
#endif
#if defined(__MSC_VER)
#define __attribute__(x) __declspec(x)
#endif



/// @brief A 2 element float vector
typedef __m128 vec2;
/// @brief A 3 element float vector
typedef __m128 vec3;
/// @brief A 4 element float vector
typedef __m128 vec4;

/// @brief A row major 2x2 float matrix
typedef __m128 mat2;
/// @brief A row major 3x3 float matrix.
/// @details The elements are stored as 3 `__m128`s. The
/// first three values are the elements and the last
/// value is undefined (not necessarily zero)
typedef struct mat3 {
    __m128 m0;
    __m128 m1;
    __m128 m2;
} __attribute__((aligned(64))) mat3;
/// @brief A row major 4x4 float matrix
typedef struct rmat4 {
    __m128 m0;
    __m128 m1;
    __m128 m2;
    __m128 m3;
} __attribute__((aligned(64))) rmat4;

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
/// where a, b, c, and d are the 4 `__m128`s
typedef struct mat4 {
    __m128 b0;
    __m128 b1;
    __m128 b2;
    __m128 b3;
} __attribute__((aligned(64))) mat4;


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
