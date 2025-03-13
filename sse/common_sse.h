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


/// @brief A 2 element float vector
typedef __m128 vec2;
/// @brief A 3 element float vector
typedef __m128 vec3;
/// @brief A 4 element float vector
typedef __m128 vec4;

/// @brief A n-element float vector stored as __m128s.
/// @details There is extra padding if n is not divisible by 4.
/// This is aligned to 64 bytes to improve cache
/// efficiency.
/// @attention The dimensions must be stored by the user. For
/// a type that handles dimension storage, consider using `vec`
typedef VECM_ALIGN_64 __m128* vecRaw;

/// @brief A row major 2x2 float matrix
typedef __m128 mat2;
/// @brief A row major 3x3 float matrix.
/// @details The elements are stored as 3 `__m128`s. The
/// first three values are the elements and the last
/// value is undefined (not necessarily zero)
typedef struct mat3 {
    /// @brief The first row of the mat3
    __m128 m0;
    /// @brief The second row of the mat3
    __m128 m1;
    /// @brief The third row of the mat3
    __m128 m2;
} VECM_ALIGN_64 mat3;
/// @brief A row major 4x4 float matrix
typedef struct rmat4 {
    /// @brief The first row of the rmat4
    __m128 m0;
    /// @brief The second row of the rmat4
    __m128 m1;
    /// @brief The third row of the rmat4
    __m128 m2;
    /// @brief The fourth row of the rmat4
    __m128 m3;
} VECM_ALIGN_64 rmat4;

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
    /// @brief The top left 2x2 block of the mat4
    __m128 b0;
    /// @brief The top right 2x2 block of the mat4
    __m128 b1;
    /// @brief The bottom left 2x2 block of the mat4
    __m128 b2;
    /// @brief The bottom right 2x2 block of the mat4
    __m128 b3;
} VECM_ALIGN_64 mat4;


/// @brief A block matrix stored as 4x4 blocks matrices further
/// subdivided into 2x2 blocks.
/// @details This (convoluted) layout is to maximize cache efficiency,
/// with 4x4 blocks aligned to the 64 bit cache lines, and a 2x2 block
/// matrix used for marginally faster computations.
/// @attention The dimensions must be stored by the user for this data
/// type. For a type that handles dimension storage, consider using `mat`
typedef VECM_ALIGN_64 __m128* matRaw;


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
