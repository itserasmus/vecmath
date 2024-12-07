/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `vec4` related functions for the AVX512 implementation of VecMath.
 */
#ifndef VEC4X4_H_AVX512
#define VEC4X4_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif


/// @brief Adds two 4-vectors.
pure_fn vec4 add_vec4(const vec4 a, const vec4 b) {
    return _mm_add_ps(a, b);
}

/// @brief Subtracts two 4-vectors.
pure_fn vec4 sub_vec4(const vec4 a, const vec4 b) {
    return _mm_sub_ps(a, b);
}

/// @brief Multiplies a 4-vector by a float.
pure_fn vec4 scalar_mul_vec4(const vec4 a, const float b) {
    return _mm_mul_ps(a, _mm_set_ps1(b));
}

/// @brief Multiplies two 4-vectors element-wise.
pure_fn vec4 mul_vec4(const vec4 a, const vec4 b) {
    return _mm_mul_ps(a, b);
}

/// @brief Divides a 4-vector by a float.
pure_fn vec4 scalar_div_vec4(const vec4 a, const float b) {
    return _mm_div_mac(a, _mm_set_ps1(b));
}

/// @brief Divides two 4-vectors element-wise.
pure_fn vec4 div_vec4(const vec4 a, const vec4 b) {
    return _mm_div_mac(a, b);
}

/// @brief Computes the dot product of two 4-vectors.
pure_fn float dot_vec4(const vec4 a, const vec4 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b11110001));
}

/// @brief Normalizes a 4-vector.
pure_fn vec4 norm_vec4(const vec4 a) {
    float len = sqrt(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b11111111)));
    return _mm_div_mac(a, _mm_set_ps1(len));
}

/// @brief Computes the length of a 4-vector.
pure_fn float len_vec4(const vec4 a) {
    return sqrt(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b11110001)));
}

/// @brief Computes the outer product of two 4-vectors as a `rmat4`
pure_fn rmat4 outer_rvec4(const vec4 a, const vec4 b) {
    rmat4 ret;
    ret.m0 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0))
        )
    );
    ret.m1 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 3, 3)),
            _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2))
        )
    );
    return ret;
}

/// @brief Computes the outer product of two 4-vectors as a `mat4`
pure_fn mat4 outer_vec4(const vec4 a, const vec4 b) {
    mat4 ret;
    ret.b0 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 2, 2)),
            _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 0, 0))
        )
    );
    ret.b1 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 3, 3)),
            _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 1, 1))
        )
    );
    
    return ret;
}






#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
