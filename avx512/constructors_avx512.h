/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains functions to create vec2-4 and mat2-4 for the AVX512 implementation of VecMath.
 */
#ifndef VEC_MATH_CONSTRUCTORS_H_AVX512
#define VEC_MATH_CONSTRUCTORS_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif


/// @brief Creates a 2-component vector from 2 float components.
/// @details This function creates a 2-component vector stored as a
/// single `__m128`. The last two elements are undefined, not
/// necessarily 0. They may be modified by VecMath and other functions
/// without warning.
pure_fn vec2 create_vec2(
    const float x, const float y) {
    return _mm_set_ps(0.0f, 0.0f, y, x);
}
/// @brief Creates a 3-component vector from 3 float components.
/// @details This function creates a 3-component vector stored as a
/// single `__m128`. The last element is undefined, not necessarily 0.
/// It may be modified by VecMath and other functions without warning.
pure_fn vec3 create_vec3(
    const float x, const float y, const float z) {
    return _mm_set_ps(0.0f, z, y, x);
}
/// @brief Creates a 4-component vector from 4 float components.
/// @details This function creates a 4-component vector stored as a
/// single `__m128`.
pure_fn vec4 create_vec4(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}

/// @brief Creates a 2x2 row-major matrix from 4 float components.
/// @details This function creates a 2x2 matrix stored as a single
/// `__m128`. The first two elements are the first row of the matrix,
/// and the last two elements are the second row of the matrix.
pure_fn mat2 create_mat2(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}


/// @brief Creates a 3x3 row-major matrix from 2 6-element vectors
/// @details This function creates a 3x3 matrix stored as 4 3-element
/// vectors (rows). The vectors (rows) are represented by the first
/// three lanes of a `__m512`. The last element of each vector and the
/// last lane is undefined, not necessarily 0. They may be modified by
/// VecMath and other functions without warning.
pure_fn mat3 create_mat3v256(
    const __m256 a, const __m256 b) {
    #ifdef __AVX512DQ__
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
    #else
    return _mm512_insertf32x4(
        _mm512_insertf32x4(_mm512_castps256_ps512(a), _mm256_castps256_ps128(b), 2),
        _mm256_extractf128_ps(b, 1), 3
    );
    #endif
}

/// @brief Creates a 3x3 row-major matrix from three 3-element vectors.
/// @details This function creates a 3x3 matrix stored as 4 3-element
/// vectors (rows). The vectors (rows) are represented by the first
/// three lanes of a `__m512`. The last element of each vector and the
/// last lane is undefined, not necessarily 0. They may be modified by
/// VecMath and other functions without warning.
pure_fn mat3 create_mat3v(
    const __m128 a, const __m128 b, const __m128 c) {
    return _mm512_insertf32x4(
        _mm512_insertf32x4(_mm512_castps128_ps512(a), b, 1),
        c, 2
    );
}

/// @brief Creates a 3x3 row-major matrix from 9 float components.
/// @details This function creates a 3x3 matrix stored as 4 3-element
/// vectors (rows). The vectors (rows) are represented by the first
/// three lanes of a `__m512`. The last element of each vector and the
/// last lane is undefined, not necessarily 0. They may be modified by
/// VecMath and other functions without warning.
pure_fn mat3 create_mat3(
    const float a, const float b, const float c,
    const float p, const float q, const float r,
    const float x, const float y, const float z) {
    return _mm512_set_ps(0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, z, y, x,
        0.0f, r, q, p,
        0.0f, c, b, a);
}

/// @brief Creates a 4x4 row-major matix from 2 8-element vectors.
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
pure_fn rmat4 create_rmat4v256(
    const __m256 a, const __m256 b) {
    #ifdef __AVX512DQ__
    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
    #else
    return _mm512_insertf32x4(
        _mm512_insertf32x4(_mm512_castps256_ps512(a),
            _mm256_castps256_ps128(b), 2),
        _mm256_extractf128_ps(b, 1), 3
    );
    #endif
}

/// @brief Creates a 4x4 row-major matix from 4 4-element vectors.
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
pure_fn rmat4 create_rmat4v(
    const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    return _mm512_insertf32x4(
        _mm512_insertf32x4(
            _mm512_insertf32x4(_mm512_castps128_ps512(a), b, 1),
            c, 2),
        d, 3
    );
}

/// @brief Creates a 4x4 row-major matix from 16 float components.
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
pure_fn rmat4 create_rmat4(
    const float a, float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    return _mm512_set_ps(w, z, y, x, s, r, q, p, k, j, i, h, d, c, b, a);
}

/// @brief Creates a 4x4 block matrix from 16 float components.
/// @attention This `mat4` is a block matrix and needs to be
/// handled accordingly when performing low level manipulation or
/// using it with other libraries. To convert a `mat4` to a row
/// major format, use the `cvt_mat4_rmat4`. Although `mat4` usually
/// performs better than the `rmat4`, for short-lived/temporary
/// matrices shared with other libraries/graphics, consider using
/// the `rmat4`.
pure_fn mat4 create_mat4(
    const float a, const float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    return _mm512_set_ps(c, d, j, k,
                         p, q, x, y,
                         a, b, h, i,
                         r, s, z, w);
}


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
