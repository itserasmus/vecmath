/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `vec3` related functions for the AVX512 implementation of VecMath.
 */
#ifndef VEC3X3_H_AVX512
#define VEC3X3_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif



/// @brief Adds two 3-vectors. 
pure_fn vec3 add_vec3(const vec3 a, const vec3 b) {
    return _mm_add_ps(a, b);
}

/// @brief Subtracts two 3-vectors 
pure_fn vec3 sub_vec3(const vec3 a, const vec3 b) {
    return _mm_sub_ps(a, b);
}

/// @brief Multiplies a 3-vector by a float.
pure_fn vec3 scalar_mul_vec3(const vec3 a, const float b) {
    return _mm_mul_ps(a, _mm_set_ps1(b));
}

/// @brief Multiplies two 3-vectors element-wise.
pure_fn vec3 mul_vec3(const vec3 a, const vec3 b) {
    return _mm_mul_ps(a, b);
}

/// @brief Divides a 3-vector by a float.
pure_fn vec3 scalar_div_vec3(const vec3 a, const float b) {
    return _mm_div_mac(a, _mm_set_ps1(b));
}

/// @brief Divides two 3-vectors element-wise.
pure_fn vec3 div_vec3(const vec3 a, const vec3 b) {
    return _mm_div_mac(a, b);
}

/// @brief Computes the dot product of two 3-vectors.
pure_fn float dot_vec3(const vec3 a, const vec3 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b01110001));
}

#ifdef VECM_RIGHT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Computes the cross product of two 3-vectors in a right handed system
pure_fn vec3 cross_vec3_right_handed(const vec3 a, const vec3 b)
#else
/// @brief Computes the cross product of two 3-vectors in a right handed system
pure_fn vec3 cross_vec3(const vec3 a, const vec3 b)
#endif
{
    return _mm_fms_mac(
        _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_permute_mac(b, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 0, 2)),
            _mm_permute_mac(b, _MM_SHUFFLE(0, 0, 2, 1))
        )
    );
}
#endif
#ifdef VECM_LEFT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Computes the cross product of two 3-vectors in a left handed system
pure_fn vec3 cross_vec3_left_handed(const vec3 a, const vec3 b)
#else
/// @brief Computes the cross product of two 3-vectors in a left handed system
pure_fn vec3 cross_vec3(const vec3 a, const vec3 b)
#endif
{
    return _mm_fms_mac(
        _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(b, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_permute_mac(b, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
}
#endif

/// @brief Normalizes a 3-vector.
pure_fn vec3 norm_vec3(const vec3 a) {
    return _mm_mul_ps(a, _mm_rsqrt_mac(_mm_dp_ps(a, a, 0b01110111)));
}

/// @brief Computes the length of a 3-vector.
pure_fn float len_vec3(const vec3 a) {
    return sqrtf(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b01110001)));
}

/// @brief Computes the outer product of two 3-vectors.
pure_fn mat3 outer_vec3(const vec3 a, const vec3 b) {
    return _mm512_mul_ps(
        _mm512_broadcast_f32x4(b),
        _mm512_insertf32x4(_mm512_insertf32x4(_mm512_castps128_ps512(
            _mm_permute_ps(a, _MM_SHUFFLE(0, 0, 0, 0))),
            _mm_permute_ps(a, _MM_SHUFFLE(1, 1, 1, 1)), 1),
            _mm_permute_ps(a, _MM_SHUFFLE(2, 2, 2, 2)), 2
        )
    );
}



void print_mat3(const mat3 a) {
    _Alignas(16) float arr[16];
    _mm512_store_ps(&arr, a);
    printf("%f %f %f\n%f %f %f\n%f %f %f\n\n",
        arr[0], arr[1], arr[2],
        arr[4], arr[5], arr[6],
        arr[8], arr[9], arr[10]
    );
}




#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
