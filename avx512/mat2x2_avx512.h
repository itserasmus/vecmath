/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat2` related functions for the AVX512 implementation of VecMath.
 */
#ifndef MAT2X2_H_AVX512
#define MAT2X2_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif



/// @brief Adds two 2x2 matrices.
pure_fn mat2 add_mat2(const mat2 a, const mat2 b) {
    return _mm_add_ps(a, b);
}

/// @brief Subtracts two 2x2 matrices.
pure_fn mat2 sub_mat2(const mat2 a, const mat2 b) {
    return _mm_sub_ps(a, b);
}

/// @brief Multiplies a 2x2 matrix by a scalar 
pure_fn mat2 scal_mul_mat2(const mat2 a, const float b) {
    return _mm_mul_ps(a, _mm_set1_ps(b));
}

/// @brief Transposes a 2x2 matrix
pure_fn mat2 trans_mat2(const mat2 a) {
    return _mm_permute_mac(a, _MM_SHUFFLE(3, 1, 2, 0));
}

/// @brief Calculates the determinant of a 2x2 matrix
pure_fn float det_mat2(const mat2 a) {
    vec4 pairs = _mm_mul_ps(a, _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3)));
    vec4 sub = _mm_sub_ps(pairs, _mm_permute_mac(pairs, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(sub);
}

/// @brief Multiplies a 2-vector by a 2x2 matrix
pure_fn vec2 mul_vec2_mat2(const vec2 a, const mat2 b) {
    __m128 prod = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 1, 0)),
        _mm_permute_mac(b, _MM_SHUFFLE(3, 1, 2, 0))
    );
    return _mm_hadd_ps(prod, prod);
}

/// @brief Multiplies a 2x2 matrix by a 2-vector
pure_fn vec2 mul_mat2_vec2(const mat2 a, const vec2 b) {
    __m128 v_mul = _mm_mul_ps(a, _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 1, 0)));
    return _mm_hadd_ps(v_mul, v_mul);
}

/// @brief Calculates the cofactor matrix of a 2x2 matrix
pure_fn mat2 cofactor_mat2(const mat2 a) {
    mat2 aperm = _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3));
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), aperm, 0b0110);
    return _mm_sub_ps(aperm, _mm_add_ps(adj_neg, adj_neg));
}

/// @brief Calculates the adjoint of a 2x2 matrix
pure_fn mat2 adj_mat2(const mat2 a) {
    mat2 aperm = _mm_permute_mac(a, _MM_SHUFFLE(0, 2, 1, 3));
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), aperm, 0b0110);
    return _mm_sub_ps(aperm, _mm_add_ps(adj_neg, adj_neg));
}

/// @brief Calculates the inverse of a 2x2 matrix
pure_fn mat2 inv_mat2(const mat2 a) {
    vec4 pairs = _mm_mul_ps(a, _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3)));
    vec4 sub = _mm_sub_ps(pairs, _mm_permute_mac(pairs, _MM_SHUFFLE(0, 0, 0, 1)));
    float det = _mm_cvtss_f32(sub);
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), a, 0b0110);
    mat2 adj_not_perm = _mm_sub_ps(a, _mm_add_ps(adj_neg, adj_neg));
    return _mm_div_mac(_mm_permute_mac(adj_not_perm, _MM_SHUFFLE(0, 2, 1, 3)), _mm_set1_ps(det));
}

/// @brief Multiplies two 2x2 matrices
pure_fn mat2 mul_mat2(const mat2 a, const mat2 b) {
    return _mm_fma_mac(
        _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 0, 0)), b,
        _mm_mul_ps(_mm_permute_mac(a, _MM_SHUFFLE(2, 2, 1, 1)), _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 3, 2)))
    );
}

/// @brief Calculates the square of a 2x2 matrix
pure_fn mat2 sqr_mat2(const mat2 a) {
    mat2 accum = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 0, 0)),
        _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 1, 0))
    );
    return _mm_fma_mac(
        _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 1, 1)),
        _mm_permute_mac(a, _MM_SHUFFLE(3, 2, 3, 2)),
        accum
    );
}

/// @brief Calculates the Nth positive integral power of a 2x2 matrix
pure_fn mat2 pow_mat2(const mat2 a, const unsigned N) {
    mat2 res = create_mat2(1.0f, 0.0f, 0.0f, 1.0f);
    mat2 base = a;
    unsigned n = N;
    
    while (n > 0) {
        if (n % 2 == 1) {
            res = mul_mat2(res, base);
        }
        base = sqr_mat2(base);
        n /= 2;
    }
    
    return res;
}

/// @brief Stores a 2x2 matrix as an array
void store_mat2(float *arr, const mat2 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

/// @brief Prints a 2x2 matrix
void print_mat2(const mat2 a) {
    printf("%f %f\n%f %f\n\n", _mm_extractf_ps(a, 0), _mm_extractf_ps(a, 1), _mm_extractf_ps(a, 2), _mm_extractf_ps(a, 3));
}

/// @brief Stores a 2-vector as an array
void store_vec2(float *arr, const vec2 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

/// @brief Prints a 2-vector
void print_vec2(const vec2 a) {
    printf("%f %f\n", _mm_extractf_ps(a, 0), _mm_extractf_ps(a, 1));
}



#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
