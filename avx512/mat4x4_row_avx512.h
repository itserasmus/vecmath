/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `rmat4` related functions for the AVX512 implementation of VecMath.
 */
#ifndef MAT4X4_ROW_H_AVX512
#define MAT4X4_ROW_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif


/// @brief Add two 4x4 matrices.
pure_fn rmat4 add_rmat4(const rmat4 a, const rmat4 b) {
    return _mm512_add_ps(a, b);
}

/// @brief Subtract two 4x4 matrices.
pure_fn rmat4 sub_rmat4(const rmat4 a, const rmat4 b) {
    return _mm512_sub_ps(a, b);
}

/// @brief Multtiply a 4x4 matrix by a scalar.
pure_fn rmat4 scal_mul_rmat4(const rmat4 a, const float b) {
    return _mm512_add_ps(a, _mm512_set1_ps(b));
}

/// @brief Transpose a 4x4 matrix.
pure_fn rmat4 trans_rmat4(const rmat4 a) {
    return _mm512_permutexvar_ps(_mm512_setr_epi32(
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15
    ), a);
}

/// @brief Compute the determinant of a 4x4 matrix.
pure_fn float det_rmat4(const rmat4 a) {
    // det = 
    // ( a c  - a c ) * ( b d  - b d )
    //   0 0    0 0       0 0    0 0
    //   0 1    1 0       3 2    2 3
    //   0 2    2 0       1 3    3 1
    //   0 3    3 0       2 1    1 2
    // ( b d  - b d ) * ( a c  - a c )
    //   0 0    0 0       0 0    0 0
    //   0 1    1 0       3 2    2 3
    //   0 2    2 0       1 3    3 1
    //   0 3    3 0       2 1    1 2
    //   C0     C1        C0     C1
    __m512 c0c1 = _mm512_sub_ps(
        _mm512_mul_ps( // C0
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                0, 0, 0, 0,
                4, 4, 4, 4,
                4, 7, 5, 6,
                0, 3, 1, 2
            ), a),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                8, 9, 10, 11,
                12, 13, 14, 15,
                12, 14, 15, 13,
                8, 10, 11, 9
            ), a)
        ),
        _mm512_mul_ps( // C1
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                0, 1, 2, 3,
                4, 5, 6, 7,
                4, 6, 7, 5,
                0, 2, 3, 1
            ), a),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                8, 8, 8, 8,
                12, 12, 12, 12,
                12, 15, 13, 14,
                8, 11, 9, 10
            ), a)
        )
    );
    // AVX512: The instruction set that dares you to mix AVX just to finish your calculations.
    __m256 prod = _mm256_dp_ps(
        _mm512_castps512_ps256(c0c1),
        #ifdef __AVX512DQ__
        _mm512_extractf32x8_ps(c0c1, 1),
        #else
        _mm256_set_m128(
            _mm512_extractf32x4_ps(c0c1, 3),
            _mm512_extractf32x4_ps(c0c1, 2)
        ),
        #endif
        0b11111111
    );
    return _mm_cvtss_f32(_mm256_castps256_ps128(prod)) + _mm_cvtss_f32(_mm256_extractf128_ps(prod, 1));
    // Gives your computer the power to set fire to your home
}

/// @brief Multiply a 4-vector by a 4x4 matrix.
pure_fn vec4 mul_vec4_rmat4(const vec4 a, const rmat4 b) {
    __m512 prod = _mm512_mul_ps(_mm512_permutexvar_ps(
        _mm512_setr_epi32(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3), _mm512_castps128_ps512(a)), b);
    return _mm_add_ps(
        _mm512_castps512_ps128(prod),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 1),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 2),
        _mm512_extractf32x4_ps(prod, 3)
    )));
}

/// @brief Multiply a 4x4 matrix by a 4-vector.
pure_fn vec4 mul_rmat4_vec4(const rmat4 a, const vec4 b) {
    __m512 prod = _mm512_permutexvar_ps(
        _mm512_setr_epi32(0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15),
        _mm512_mul_ps(_mm512_broadcast_f32x4(b), a));
    return _mm_add_ps(
        _mm512_castps512_ps128(prod),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 1),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 2),
        _mm512_extractf32x4_ps(prod, 3)
    )));
}

/// @brief Multiply two 4x4 matrices.
pure_fn rmat4 mul_rmat4(const rmat4 a, const rmat4 b) {
    return _mm512_fma_mac(
        _mm512_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm512_permutexvar_ps(_mm512_setr4_epi32(0, 1, 2, 3), b),
        _mm512_fma_mac(
            _mm512_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm512_permutexvar_ps(_mm512_setr4_epi32(4, 5, 6, 7), b),
            _mm512_fma_mac(
                _mm512_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm512_permutexvar_ps(_mm512_setr4_epi32(8, 9, 10, 11), b),
                _mm512_mul_ps(
                    _mm512_permute_mac(a, _MM_SHUFFLE(3, 3, 3, 3)),
                    _mm512_permutexvar_ps(_mm512_setr4_epi32(12, 13, 14, 15), b)
                )
            )
        )
    );
}

/// @brief Compute the cofactor of a 4x4 matrix.
pure_fn rmat4 cofactor_rmat4(const rmat4 a) {
    rmat4 ret;
    // pain pain pain
    __m512 dets = _mm512_fmsub_ps(
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            9, 8, 8, 9,
            11, 10, 9, 8,
            1, 0, 0, 1,
            3, 2, 1, 0
        ), a),
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            15, 14, 15, 14,
            14, 15, 12, 13,
            7, 6, 7, 6,
            6, 7, 4, 5
        ), a),
        _mm512_mul_ps(
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                11, 10, 11, 10,
                10, 11, 8, 9,
                3, 2, 3, 2,
                2, 3, 0, 1
            ), a),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                13, 12, 12, 13,
                15, 14, 13, 12,
                5, 4, 4, 5,
                7, 6, 5, 4
            ), a)
        )
    );
    // cuz somebody forgot _mm512_permute4f128_ps
    __m512 a_rowswap = _mm512_permutexvar_ps(_mm512_setr_epi32(
        7, 6, 4, 5,
        3, 2, 0, 1,
        15, 14, 12, 13,
        11, 10, 8, 9
    ), a);
    ret = _mm512_permutexvar_ps(_mm512_setr_epi32( // dets_0022 (reuse variable)
        0, 1, 2, 3,
        0, 1, 2, 3,
        8, 9, 10, 11,
        8, 9, 10, 11
    ), dets);


    ret = _mm512_fmsub_ps(
        a_rowswap,
        _mm512_permute_ps(ret, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm512_fmadd_ps(
            _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                4, 5, 6, 7,
                4, 5, 6, 7,
                12, 13, 14, 15,
                12, 13, 14, 15
            ), dets),
            _mm512_mul_ps(
                _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(2, 3, 0, 1)),
                ret
            )
        )
    );
    dets = _mm512_setr_ps( // used as a XOR mask for negation
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f,
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f
    );
    
    // cast to and from __m512i cuz Intel
    // forgot to add _mm512_xor_ps, among
    // numerous other things...
    ret = _mm512_castsi512_ps(
        _mm512_xor_epi32(
            _mm512_castps_si512(ret),
            _mm512_castps_si512(dets)
        )
    );
    

    return ret;
}

/// @brief Compute the adjoint of a 4x4 matrix.
pure_fn rmat4 adj_rmat4(const rmat4 a) {
    // for logic refer to the AVX implementation of `cofactor_rmat4`
    rmat4 ret;
    // same stuff as AVX512 `cofactor_rmat4`
    __m512 dets = _mm512_fms_mac(
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            6, 2, 6, 2,
            10, 11, 2, 3,
            4, 0, 4, 0,
            8, 9, 0, 1
        ), a),
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            11, 15, 15, 11,
            15, 14, 7, 6,
            9, 13, 13, 9,
            13, 12, 5, 4
        ), a),
        _mm512_mul_ps(
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                7, 3, 7, 3,
                11, 10, 3, 2,
                5, 1, 5, 1,
                9, 8, 1, 0
            ), a),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                10, 14, 14, 10,
                14, 15, 6, 7,
                8, 12, 12, 8,
                12, 13, 4, 5
            ), a)
        )
    );
    __m512 a_rowswap = _mm512_permutexvar_ps(_mm512_setr_epi32(
        13, 9, 1, 5,
        12, 8, 0, 4,
        15, 11, 3, 7,
        14, 10, 2, 6
    ), a);
    ret = _mm512_permutexvar_ps(_mm512_setr_epi32( // dets_0022 (reuse variable)
        0, 1, 2, 3,
        0, 1, 2, 3,
        8, 9, 10, 11,
        8, 9, 10, 11
    ), dets);


    ret = _mm512_fma_mac(
        a_rowswap, // 3 2 0 1
        ret,
        _mm512_fms_mac(
            _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                4, 5, 6, 7,
                4, 5, 6, 7,
                12, 13, 14, 15,
                12, 13, 14, 15
            ), dets),
            _mm512_mul_ps(
                _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(2, 3, 0, 1)),
                _mm512_permute_ps(ret, _MM_SHUFFLE(0, 1, 3, 2))
            )
        )
    );
    dets = _mm512_setr_ps(
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f,
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f
    );

    ret = _mm512_castsi512_ps(
        _mm512_xor_epi32(
            _mm512_castps_si512(ret),
            _mm512_castps_si512(dets)
        )
    );
    
    return ret;
}

/// @brief Compute the inverse of a 4x4 matrix.
pure_fn rmat4 inv_rmat4(const rmat4 a) {
    // for logic refer to the AVX implementation of `adj_rmat4`
    rmat4 ret;
    // same stuff as AVX512 `cofactor_rmat4`
    __m512 dets = _mm512_fms_mac(
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            6, 2, 6, 2,
            10, 11, 2, 3,
            4, 0, 4, 0,
            8, 9, 0, 1
        ), a),
        _mm512_permutexvar_ps(_mm512_setr_epi32(
            11, 15, 15, 11,
            15, 14, 7, 6,
            9, 13, 13, 9,
            13, 12, 5, 4
        ), a),
        _mm512_mul_ps(
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                7, 3, 7, 3,
                11, 10, 3, 2,
                5, 1, 5, 1,
                9, 8, 1, 0
            ), a),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                10, 14, 14, 10,
                14, 15, 6, 7,
                8, 12, 12, 8,
                12, 13, 4, 5
            ), a)
        )
    );
    __m512 a_rowswap = _mm512_permutexvar_ps(_mm512_setr_epi32(
        13, 9, 1, 5,
        12, 8, 0, 4,
        15, 11, 3, 7,
        14, 10, 2, 6
    ), a);
    ret = _mm512_permutexvar_ps(_mm512_setr_epi32( // dets_0022 (reuse variable)
        0, 1, 2, 3,
        0, 1, 2, 3,
        8, 9, 10, 11,
        8, 9, 10, 11
    ), dets);


    ret = _mm512_fma_mac(
        a_rowswap, // 3 2 0 1
        ret,
        _mm512_fms_mac(
            _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm512_permutexvar_ps(_mm512_setr_epi32(
                4, 5, 6, 7,
                4, 5, 6, 7,
                12, 13, 14, 15,
                12, 13, 14, 15
            ), dets),
            _mm512_mul_ps(
                _mm512_permute_ps(a_rowswap, _MM_SHUFFLE(2, 3, 0, 1)),
                _mm512_permute_ps(ret, _MM_SHUFFLE(0, 1, 3, 2))
            )
        )
    );
    dets = _mm512_setr_ps(
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f,
        0.0f,0.0f,0.0f,0.0f,
        -0.0f,-0.0f,-0.0f,-0.0f
    );

    ret = _mm512_castsi512_ps(
        _mm512_xor_epi32(
            _mm512_castps_si512(ret),
            _mm512_castps_si512(dets)
        )
    );
    __m256 determinant = _mm256_rcp_mac(_mm256_dp_ps(_mm512_castps512_ps256(_mm512_permutexvar_ps(_mm512_setr_epi32(
        0, 4, 8, 12,
        1, 5, 9, 13, /* junk values*/ 0,0,0,0, 0,0,0,0
    ), ret)), _mm512_castps512_ps256(a), 0b11111111));

    ret = _mm512_mul_ps(
        ret,
        #ifdef __AVX512DQ__
        _mm512_insertf32x8(_mm512_castps256_ps512(dets), dets, 1);
        #else
        _mm512_insertf32x4(
            _mm512_insertf32x4(_mm512_castps256_ps512(determinant), _mm256_castps256_ps128(determinant), 2),
            _mm256_extractf128_ps(determinant, 1), 3
        )
        #endif
    );
    
    return ret;
}

/// @brief Compute the square of a 4x4 matrix.
pure_fn rmat4 sqr_rmat4(const rmat4 a) {
    return _mm512_fma_mac(
        _mm512_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm512_permutexvar_ps(_mm512_setr4_epi32(0, 1, 2, 3), a),
        _mm512_fma_mac(
            _mm512_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm512_permutexvar_ps(_mm512_setr4_epi32(4, 5, 6, 7), a),
            _mm512_fma_mac(
                _mm512_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm512_permutexvar_ps(_mm512_setr4_epi32(8, 9, 10, 11), a),
                _mm512_mul_ps(
                    _mm512_permute_mac(a, _MM_SHUFFLE(3, 3, 3, 3)),
                    _mm512_permutexvar_ps(_mm512_setr4_epi32(12, 13, 14, 15), a)
                )
            )
        )
    );
}

/// @brief Compute the Nth positive integral power of a 4x4 matrix.
pure_fn rmat4 pow_rmat4(const rmat4 a, const unsigned N) {
    rmat4 res = create_rmat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    rmat4 base = a;
    unsigned n = N;
    
    while (n > 0) {
        if (n % 2 == 1) {
            res = mul_rmat4(res, base);
        }
        base = sqr_rmat4(base);
        n /= 2;
    }
    
    return res;
}




/// @brief Store a 4x4 matrix as an array.
void store_rmat4(float* arr, const rmat4 a) { // arr must be at least 16 wide
    _mm512_storeu_ps(arr, a);
}

#ifdef __cplusplus
#define _Alignas(x)
#endif
/// @brief Print a 4x4 matrix.
void print_rmat4(const rmat4 a) {
    _Alignas(16) float arr[16];
    _mm512_store_ps(arr, a);
    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        arr[0], arr[1], arr[2], arr[3],
        arr[4], arr[5], arr[6], arr[7],
        arr[8], arr[9], arr[10], arr[11],
        arr[12], arr[13], arr[14], arr[15]
    );
}

/// @brief Store a 4-vector as an array.
void store_vec4(float* arr, const vec4 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

/// @brief Print a 4-vector.
void print_vec4(const vec4 a) {
    printf("%f %f %f %f\n",
        _mm_extractf_ps(a, 0),
        _mm_extractf_ps(a, 1),
        _mm_extractf_ps(a, 2),
        _mm_extractf_ps(a, 3)
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
