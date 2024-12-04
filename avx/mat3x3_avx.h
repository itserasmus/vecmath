/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat3` related functions for the AVX implementation of VecMath.
 */
#ifndef MAT3X3_H_AVX
#define MAT3X3_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif


/// @brief Adds two 3x3 matrices.
pure_fn mat3 add_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm256_add_ps(a.m0, b.m0);
    ret.m1 = _mm_add_ps(a.m1, b.m1);
    return ret;
}

/// @brief Subtracts two 3x3 matrices.
pure_fn mat3 sub_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm256_sub_ps(a.m0, b.m0);
    ret.m1 = _mm_sub_ps(a.m1, b.m1);
    return ret;
}

/// @brief Multiplies a 3x3 matrix by a scalar.
pure_fn mat3 scal_mul_mat3(const mat3 a, const float b) {
    mat3 ret;
    ret.m0 = _mm256_mul_ps(a.m0, _mm256_set1_ps(b));
    ret.m1 = _mm_mul_ps(a.m1, _mm_set_ps1(b));
    return ret;
}

/// @brief Transposes a 3x3 matrix.
pure_fn mat3 trans_mat3(const mat3 a) {
    mat3 ret;
    __m128 tmp = _mm_movelh_ps( // contains m0 -> 0 1 4 5
        _mm256_castps256_ps128(a.m0),
        _mm256_extractf128_ps(a.m0, 1)
    );
    ret.m0 = _mm256_set_m128(
        _mm_shuffle_ps(tmp, a.m1, _MM_SHUFFLE(0, 1, 3, 1)),
        _mm_shuffle_ps(tmp, a.m1, _MM_SHUFFLE(0, 0, 2, 0))
    );
    tmp = _mm_movehl_ps( // contains m0 -> 2 3 6 7
        _mm256_castps256_ps128(a.m0),
        _mm256_extractf128_ps(a.m0, 1)
    );
    ret.m1 = _mm_shuffle_ps(tmp, a.m1, _MM_SHUFFLE(0, 2, 0, 2));
    return ret;
}

/// @brief Computes the determinant of a 3x3 matrix.
pure_fn float det_mat3(const mat3 a) {
    // det = 
    // c0 * (b1a2 - b2a1) +
    // c1 * (b2a0 - b0a2) +
    // c2 * (b0a1 - b1a0)
    // m0 * (C1C0  -C0C1)
    __m256 c0c1 = _mm256_mul_ps(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm256_permute_mac(
            _mm256_permute2f128_ps(a.m0, a.m0, 0b01),
            _MM_SHUFFLE(0, 0, 2, 1)
        )
    );
    __m128 diff = _mm_sub_ps(
        _mm256_extractf128_ps(c0c1, 1),
        _mm256_castps256_ps128(c0c1)
    );
    return _mm_cvtss_f32(_mm_dp_ps(a.m1, diff, 0b01110001));
}

/// @brief Multiplies a 3-vector by a 3x3 matrix.
pure_fn vec3 mul_vec3_mat3(const vec3 a, const mat3 b) {
    __m128 tmp = _mm_movelh_ps( // contains m0 -> 0 1 4 5
        _mm256_castps256_ps128(b.m0),
        _mm256_extractf128_ps(b.m0, 1)
    );
    __m128 tmp2 = _mm_movehl_ps( // contains m0 -> 2 3 6 7
        _mm256_castps256_ps128(b.m0),
        _mm256_extractf128_ps(b.m0, 1)
    );
    return create_vec3(
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(tmp, b.m1, _MM_SHUFFLE(0, 0, 2, 0)), 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(tmp, b.m1, _MM_SHUFFLE(0, 1, 3, 1)), 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(tmp2, b.m1, _MM_SHUFFLE(0, 2, 0, 2)), 0b01110001))
    );
}

/// @brief Multiplies a 3x3 matrix by a 3-vector.
pure_fn vec3 mul_mat3_vec3(const mat3 a, const vec3 b) {
    return create_vec3(
        _mm_cvtss_f32(_mm_dp_ps(_mm256_castps256_ps128(a.m0), b, 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(_mm256_extractf128_ps(a.m0, 1), b, 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m1, b, 0b01110001))
    );
}

/// @brief Multiplies two 3x3 matrices.
pure_fn mat3 mul_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm256_permute2f128_ps(b.m0, b.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm256_permute2f128_ps(b.m0, b.m0, 0b00010001),
            _mm256_mul_ps(
                _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm256_set_m128(b.m1, b.m1)
            )
        )
    );
    __m256 tmp = _mm256_mul_ps(
        _mm256_set_m128(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 2, 2, 2)),
            _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 1, 1, 1))
        ),
        _mm256_permute2f128_ps(b.m0, _mm256_castps128_ps256(b.m1), 0b00100001)
    );
    ret.m1 = _mm_fma_mac(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm256_castps256_ps128(b.m0),
        _mm_add_ps(
            _mm256_castps256_ps128(tmp),
            _mm256_extractf128_ps(tmp, 1)
        )
    );
    return ret;
}

/// @brief Computes the cofator of a 3x3 matrix.
pure_fn mat3 cofactor_mat3(const mat3 a) {
    mat3 ret;
    __m256 tmp = _mm256_set_m128(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))
    );
    __m256 perm1 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1));
    __m256 perm2 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2));
    ret.m0 = _mm256_fms_mac(
        tmp, _mm256_permute2f128_ps(perm1, perm2, 0b00100001),
        _mm256_mul_ps(
            _mm256_permute2f128_ps(tmp, tmp, 0b00000001),
            _mm256_permute2f128_ps(perm1, perm2, 0b00000011)
        )
    );
    tmp = _mm256_mul_ps(
        _mm256_permute2f128_ps(perm1, perm2, 0b00100000),
        _mm256_permute2f128_ps(perm1, perm2, 0b00010011)
    );
    ret.m1 = _mm_sub_ps(
        _mm256_castps256_ps128(tmp),
        _mm256_extractf128_ps(tmp, 1)
    );
    return ret;
}

/// @brief Computes the adjoint of a 3x3 matrix.
pure_fn mat3 adj_mat3(const mat3 a) {
    mat3 store;
    __m256 tmp = _mm256_set_m128(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))
    );
    __m256 perm1 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1));
    __m256 perm2 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2));
    store.m0 = _mm256_fms_mac(
        tmp, _mm256_permute2f128_ps(perm1, perm2, 0b00100001),
        _mm256_mul_ps(
            _mm256_permute2f128_ps(tmp, tmp, 0b00000001),
            _mm256_permute2f128_ps(perm1, perm2, 0b00000011)
        )
    );
    tmp = _mm256_mul_ps(
        _mm256_permute2f128_ps(perm1, perm2, 0b00100000),
        _mm256_permute2f128_ps(perm1, perm2, 0b00010011)
    );
    store.m1 = _mm_sub_ps(
        _mm256_castps256_ps128(tmp),
        _mm256_extractf128_ps(tmp, 1)
    );
    __m128 tmp2 = _mm_movelh_ps( // contains m0 -> 0 1 4 5
        _mm256_castps256_ps128(store.m0),
        _mm256_extractf128_ps(store.m0, 1)
    );
    mat3 ret;
    ret.m0 = _mm256_set_m128(
        _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 1, 3, 1)),
        _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 0, 2, 0))
    );
    tmp2 = _mm_movehl_ps( // contains m0 -> 2 3 6 7
        _mm256_castps256_ps128(store.m0),
        _mm256_extractf128_ps(store.m0, 1)
    );
    ret.m1 = _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 2, 0, 2));
    return ret;
}

/// @brief Computes the inverse of a 3x3 matrix.
pure_fn mat3 inv_mat3(const mat3 a) {
    mat3 store;
    __m256 tmp = _mm256_set_m128(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))
    );
    __m256 perm1 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1));
    __m256 perm2 = _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2));
    store.m0 = _mm256_fms_mac(
        tmp, _mm256_permute2f128_ps(perm1, perm2, 0b00100001),
        _mm256_mul_ps(
            _mm256_permute2f128_ps(tmp, tmp, 0b00000001),
            _mm256_permute2f128_ps(perm1, perm2, 0b00000011)
        )
    );
    tmp = _mm256_mul_ps(
        _mm256_permute2f128_ps(perm1, perm2, 0b00100000),
        _mm256_permute2f128_ps(perm1, perm2, 0b00010011)
    );
    store.m1 = _mm_sub_ps(
        _mm256_castps256_ps128(tmp),
        _mm256_extractf128_ps(tmp, 1)
    );
    tmp = _mm256_rcp_mac(_mm256_set1_ps(_mm_cvtss_f32(_mm_dp_ps(a.m1, store.m1, 0b01110001))));
    __m128 tmp2 = _mm_movelh_ps( // contains m0 -> 0 1 4 5
        _mm256_castps256_ps128(store.m0),
        _mm256_extractf128_ps(store.m0, 1)
    );
    mat3 ret;
    ret.m0 = _mm256_mul_ps(
        tmp,
        _mm256_set_m128(
            _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 1, 3, 1)),
            _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 0, 2, 0))
        )
    );
    tmp2 = _mm_movehl_ps( // contains m0 -> 2 3 6 7
        _mm256_castps256_ps128(store.m0),
        _mm256_extractf128_ps(store.m0, 1)
    );
    ret.m1 = _mm_mul_ps(_mm256_castps256_ps128(tmp), _mm_shuffle_ps(tmp2, store.m1, _MM_SHUFFLE(0, 2, 0, 2)));
    return ret;
}

/// @brief Computes the square of a 3x3 matrix.
pure_fn mat3 sqr_mat3(const mat3 a) {
    mat3 ret;
    ret.m0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm256_permute2f128_ps(a.m0, a.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm256_permute2f128_ps(a.m0, a.m0, 0b00010001),
            _mm256_mul_ps(
                _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm256_set_m128(a.m1, a.m1)
            )
        )
    );
    __m256 tmp = _mm256_mul_ps(
        _mm256_set_m128(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 2, 2, 2)),
            _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 1, 1, 1))
        ),
        _mm256_permute2f128_ps(a.m0, _mm256_castps128_ps256(a.m1), 0b00100001)
    );
    ret.m1 = _mm_fma_mac(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm256_castps256_ps128(a.m0),
        _mm_add_ps(
            _mm256_castps256_ps128(tmp),
            _mm256_extractf128_ps(tmp, 1)
        )
    );
    return ret;
}


/// @brief Computes the Nth positive integral power of a 3x3 matrix.
pure_fn mat3 pow_mat3(const mat3 a, const unsigned N) {
    mat3 res = create_mat3(1,0,0, 0,1,0, 0,0,1);
    mat3 base = a;
    unsigned n = N;
    
    while (n > 0) {
        if (n % 2 == 1) {
            res = mul_mat3(res, base);
        }
        base = sqr_mat3(base);
        n /= 2;
    }
    
    return res;
}



/// @brief Stores a 3x3 matrix as an array.
void store_mat3(float *arr, const mat3 a) { // arr must be at least 10 wide
    _mm256_storeu_ps(arr, a.m0);
    _mm_storeu_ps(arr + 6, a.m1);
}

/// @brief Prints a 3x3 matrix.
void print_mat3(const mat3 a) {
    _Alignas(16) float a_arr[12];
    _mm256_store_ps(a_arr, a.m0);
    _mm_store_ps(a_arr + 8, a.m1);
    printf("%f %f %f\n%f %f %f\n%f %f %f\n\n",
        a_arr[0], a_arr[1], a_arr[2],
        a_arr[4], a_arr[5], a_arr[6],
        a_arr[8], a_arr[9], a_arr[10]
    );
}

/// @brief Stores a vec3 as an array.
void store_vec3(float *arr, const vec3 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

/// @brief Prints a vec3.
void print_vec3(const vec3 a) {
    _Alignas(16) float a_arr[4];
    _mm_store_ps(a_arr, a);
    printf("%f %f %f\n", a_arr[0], a_arr[1], a_arr[2]);
}


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
