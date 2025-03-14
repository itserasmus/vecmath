/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat3` related functions for the SSE implementation of VecMath.
 */
#ifndef MAT3X3_H_SSE
#define MAT3X3_H_SSE
#include "vec_math_sse.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif


/// @brief Adds two 3x3 matrices.
pure_fn mat3 add_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm_add_ps(a.m0, b.m0);
    ret.m1 = _mm_add_ps(a.m1, b.m1);
    ret.m2 = _mm_add_ps(a.m2, b.m2);
    return ret;
}

/// @brief Subtracts two 3x3 matrices.
pure_fn mat3 sub_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm_sub_ps(a.m0, b.m0);
    ret.m1 = _mm_sub_ps(a.m1, b.m1);
    ret.m2 = _mm_sub_ps(a.m2, b.m2);
    return ret;
}

/// @brief Multiplies a 3x3 matrix by a scalar.
pure_fn mat3 scal_mul_mat3(const mat3 a, const float b) {
    __m128 b_vec = _mm_set_ps1(b);
    mat3 ret;
    ret.m0 = _mm_mul_ps(a.m0, b_vec);
    ret.m1 = _mm_mul_ps(a.m1, b_vec);
    ret.m2 = _mm_mul_ps(a.m2, b_vec);
    return ret;
}

/// @brief Transposes a 3x3 matrix.
pure_fn mat3 trans_mat3(const mat3 a) {
    __m128 _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(a.m0, a.m1, 0x44);
    _Tmp1 = _mm_shuffle_ps(a.m2, a.m1, 0x44);
    mat3 ret;
    ret.m0 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
    ret.m1 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
    ret.m2 = _mm_shuffle_ps(_mm_shuffle_ps(a.m0, a.m1, 0xEE), _mm_shuffle_ps(a.m2, a.m1, 0xEE), 0x88);
    return ret;
}

/// @brief Computes the determinant of a 3x3 matrix.
pure_fn float det_mat3(const mat3 a) {
    // det = 
    // a0 * (b1c2 - b2c1) +
    // a1 * (b2c0 - b0c2) +
    // a2 * (b0c1 - b1c0)
    // m0 * ( C0  -  C2 )
    __m128 c0 = _mm_mul_ps(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2))
    );
    __m128 c1 = _mm_mul_ps(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1))
    );
    return _mm_cvtss_f32(_mm_dp_ps(a.m0, _mm_sub_ps(c0, c1), 0b01110001));
}

/// @brief Multiplies a 3-vector by a 3x3 matrix.
pure_fn vec3 mul_vec3_mat3(const vec3 a, const mat3 b) {
    __m128 _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(b.m0, b.m1, 0x44);
    _Tmp1 = _mm_shuffle_ps(b.m2, b.m1, 0x44);
    return create_vec3(
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88), 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a,  _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD), 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_mm_shuffle_ps(b.m0, b.m1, 0xEE), _mm_shuffle_ps(b.m2, b.m1, 0xEE), 0x88), 0b01110001))
    );
}

/// @brief Multiplies a 3x3 matrix by a 3-vector.
pure_fn vec3 mul_mat3_vec3(const mat3 a, const vec3 b) {
    return create_vec3(
        _mm_cvtss_f32(_mm_dp_ps(a.m0, b, 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m1, b, 0b01110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m2, b, 0b01110001))
    );
}

/// @brief Multiplies two 3x3 matrices.
pure_fn mat3 mul_mat3(const mat3 a, const mat3 b) {
    mat3 ret;
    ret.m0 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 2))), b.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 0))), b.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 1))), b.m1)
        )
    );
    ret.m1 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 2))), b.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 0))), b.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 1))), b.m1)
        )
    );
    ret.m2 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 2))), b.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 0))), b.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 1))), b.m1)
        )
    );
    return ret;
}

/// @brief Computes the cofactor of a 3x3 matrix.
pure_fn mat3 cofactor_mat3(const mat3 a) {
    mat3 ret;
    ret.m0 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    ret.m1 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    ret.m2 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    return ret;
}

/// @brief Computes the adjoint of a 3x3 matrix.
pure_fn mat3 adj_mat3(const mat3 a) {
    mat3 ret;
    ret.m0 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    ret.m1 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    ret.m2 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    
    __m128 _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(ret.m0, ret.m1, 0x44);
    _Tmp1 = _mm_shuffle_ps(ret.m2, ret.m1, 0x44);

    mat3 ret_adj;
    ret_adj.m0 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
    ret_adj.m1 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
    ret_adj.m2 = _mm_shuffle_ps(_mm_shuffle_ps(ret.m0, ret.m1, 0xEE), _mm_shuffle_ps(ret.m2, ret.m1, 0xEE), 0x88);
    return ret_adj;
}

/// @brief Computes the inverse of a 3x3 matrix.
pure_fn mat3 inv_mat3(const mat3 a) {
    __m128 det_arr = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    __m128 inv_det = _mm_rcp_mac(_mm_dp_ps(det_arr, a.m0, 0b01111111));
    mat3 ret;
    ret.m0 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    ret.m1 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 2, 1))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    ret.m2 = _mm_sub_ps(
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2))),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 2, 1)))
    );
    
    __m128 _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(ret.m0, ret.m1, 0x44);
    _Tmp1 = _mm_shuffle_ps(ret.m2, ret.m1, 0x44);

    mat3 ret_inv;
    ret_inv.m0 = _mm_mul_ps(inv_det, _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88));
    ret_inv.m1 = _mm_mul_ps(inv_det, _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD));
    ret_inv.m2 = _mm_mul_ps(inv_det, _mm_shuffle_ps(_mm_shuffle_ps(ret.m0, ret.m1, 0xEE), _mm_shuffle_ps(ret.m2, ret.m1, 0xEE), 0x88));
    return ret_inv;
}


/// @brief Computes the square of a 3x3 matrix.
pure_fn mat3 sqr_mat3(const mat3 a) {
    mat3 ret;
    ret.m0 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 2))), a.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 0))), a.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 1))), a.m1)
        )
    );
    ret.m1 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 2))), a.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 0))), a.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 1))), a.m1)
        )
    );
    ret.m2 = _mm_fma_mac(
        _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 2))), a.m2,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 0))), a.m0,
            _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 1))), a.m1)
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


/// @brief Stores a mat3 as an array.
void store_mat3(float *arr, const mat3 a) { // arr must be at least 10 wide
    _mm_storeu_ps(arr, a.m0);
    _mm_storeu_ps(arr + 3, a.m1);
    _mm_storeu_ps(arr + 6, a.m2);
}

/// @brief Prints a mat3.
void print_mat3(const mat3 a) {
    _Alignas(16) float a_arr[12];
    _mm_store_ps(a_arr, a.m0);
    _mm_store_ps(a_arr + 4, a.m1);
    _mm_store_ps(a_arr + 8, a.m2);
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
