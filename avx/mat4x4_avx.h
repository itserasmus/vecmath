/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat4` related functions for the AVX implementation of VecMath.
 */
#ifndef MAT4X4_H_AVX
#define MAT4X4_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif

// add, sub, scal_mul, mul, det, adj, inv, trans, pre_vec_mul, post_vec_mul, powers


/// @brief Add two 4x4 matrices.
pure_fn mat4 add_mat4(const mat4 a, const mat4 b) {
    mat4 ret;
    ret.b0 = _mm256_add_ps(a.b0, b.b0);
    ret.b1 = _mm256_add_ps(a.b1, b.b1);
    return ret;
}

/// @brief Subtract two 4x4 matrices.
pure_fn mat4 sub_mat4(const mat4 a, const mat4 b) {
    mat4 ret;
    ret.b0 = _mm256_sub_ps(a.b0, b.b0);
    ret.b1 = _mm256_sub_ps(a.b1, b.b1);
    return ret;
}

/// @brief Multiply a 4x4 matrix by a scalar.
pure_fn mat4 scal_mul_mat4(const mat4 a, const float b) {
    mat4 ret;
    __m256 b_vec = _mm256_set1_ps(b);
    ret.b0 = _mm256_mul_ps(a.b0, b_vec);
    ret.b1 = _mm256_mul_ps(a.b1, b_vec);
    return ret;
}

/// @brief Transpose a 4x4 matrix.
pure_fn mat4 trans_mat4(const mat4 a) {
    mat4 ret;
    ret.b0 = _mm256_permute_mac(a.b0, _MM_SHUFFLE(3, 1, 2, 0));
    ret.b1 = _mm256_permute2f128_ps(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(3, 1, 2, 0)),
        ret.b0,
        0b00000001
    );
    return ret;
}

/// @brief Compute the determinant of a 4x4 matrix.
pure_fn float det_mat4(const mat4 a) {
    // det =
    // (a0a3 - a1a2)(c0c3 - c1c2) +
    // (b0b3 - b1b2)(d0d3 - d1d2) +
    // 
    // ( ab  -  ab )( cd  -  cd )
    //   0 3   2 1    1 2   3 0
    //   1 2   3 0    0 3   2 1
    //   2 0   0 2    1 3   3 1
    //   3 1   1 3    0 2   2 0
    // ( cd  -  cd )
    //   1 2   3 0
    //   0 3   2 1
    //   1 3   3 1
    //   0 2   2 0
    __m256 prods_diff = _mm256_sub_ps(
        _mm256_mul_ps(
            _mm256_setr_m128(
                _mm_permute_mac(_mm256_castps256_ps128(a.b0), _MM_SHUFFLE(3, 2, 1, 0)),
                _mm_permute_mac(_mm256_extractf128_ps(a.b1, 1), _MM_SHUFFLE(0, 1, 0, 1))
            ),
            _mm256_setr_m128(
                _mm_permute_mac(_mm256_castps256_ps128(a.b1), _MM_SHUFFLE(1, 0, 2, 3)),
                _mm_permute_mac(_mm256_extractf128_ps(a.b0, 1), _MM_SHUFFLE(2, 3, 3, 2))
            )
        ),
        _mm256_mul_ps(
            _mm256_setr_m128(
                _mm_permute_mac(_mm256_castps256_ps128(a.b0), _MM_SHUFFLE(1, 0, 3, 2)),
                _mm_permute_mac(_mm256_extractf128_ps(a.b1, 1), _MM_SHUFFLE(2, 3, 2, 3))
            ),
            _mm256_setr_m128(
                _mm_permute_mac(_mm256_castps256_ps128(a.b1), _MM_SHUFFLE(3, 2, 0, 1)),
                _mm_permute_mac(_mm256_extractf128_ps(a.b0, 1), _MM_SHUFFLE(0, 1, 1, 0))
            )
        )
    );
    float lower = _mm_cvtss_f32(_mm_dp_ps(
        _mm256_castps256_ps128(prods_diff),
        _mm256_extractf128_ps(prods_diff, 1),
        0b11110001
    ));
    prods_diff = _mm256_mul_ps(
        _mm256_blend_ps(a.b0, a.b1, 0b11001100),
        _mm256_permute_mac(_mm256_blend_ps(a.b0, a.b1, 0b00110011), _MM_SHUFFLE(0, 1, 2, 3))
    );
    __m128 prods = _mm_hsub_ps(_mm256_castps256_ps128(prods_diff), _mm256_extractf128_ps(prods_diff, 1));
    float upper = _mm_cvtss_f32(_mm_dp_ps(prods, _mm_permute_mac(prods, _MM_SHUFFLE(1, 0, 3, 2)), 0b00110001));
    return lower + upper;
}

/// @brief Multiply a 4-vector with a 4x4 matrix.
pure_fn vec4 mul_vec4_mat4(const vec4 a, const mat4 b) {
    __m256 a_exp = _mm256_set_m128(_mm_permute_mac(a, _MM_SHUFFLE(3, 2, 3, 2)), _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 1, 0)));
    __m256 prod = _mm256_fma_mac(
        a_exp, _mm256_permute_mac(b.b0, _MM_SHUFFLE(3, 1, 2, 0)),
        _mm256_permute2f128_ps(_mm256_mul_ps(a_exp, _mm256_permute_mac(b.b1, _MM_SHUFFLE(3, 1, 2, 0))), b.b1, 0b00000001)
    );
    return _mm_hadd_ps(_mm256_castps256_ps128(prod), _mm256_extractf128_ps(prod, 1));
}

/// @brief Multiply a 4x4 matrix with a 4-vector.
pure_fn vec4 mul_mat4_vec4(const mat4 a, const vec4 b) {
    __m256 b_exp = _mm256_set_m128(_mm_permute_mac(b, _MM_SHUFFLE(3, 2, 3, 2)), _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 1, 0)));
    __m256 prod = _mm256_fma_mac(
        b_exp, a.b0,
        _mm256_mul_ps(_mm256_permute2f128_ps(b_exp, b_exp, 0b00000001), a.b1)
    );
    return _mm_hadd_ps(_mm256_castps256_ps128(prod), _mm256_extractf128_ps(prod, 1));
}

/// @brief Multiply two 4x4 matrices.
pure_fn mat4 mul_mat4(const mat4 a, const mat4 b) {
    // A01, B01 are the two __m256s of a
    // C01, D01 are the two __m256s of b
    // C10, D10 are the two __m256s of b with swapped lanes
    // multiply the "blocks" like 2x2 matrices, not pairwise
    // ret.b0 = A01*C01 + B01*D10
    // ret.b1 = A01*D01 + B01*C10
    mat4 ret;
    ret.b1 = _mm256_permute2f128_ps(b.b1, b.b1, 0b00000001); // use as temporary variable
    ret.b0 = _mm256_fma_mac(
        _mm256_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0)), b.b0,
        _mm256_fma_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1)),
            _mm256_permute_mac(b.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_fma_mac(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0)), ret.b1,
                _mm256_mul_ps(
                    _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1)),
                    _mm256_permute_mac(ret.b1, _MM_SHUFFLE(1, 0, 3, 2))
                )
        )
        )
    );
    ret.b1 = _mm256_permute2f128_ps(b.b0, b.b0, 0b00000001); // use as temporary variable
    ret.b1 = _mm256_fma_mac(
        _mm256_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0)), b.b1,
        _mm256_fma_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1)),
            _mm256_permute_mac(b.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_fma_mac(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0)), ret.b1,
                _mm256_mul_ps(
                    _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1)),
                    _mm256_permute_mac(ret.b1, _MM_SHUFFLE(1, 0, 3, 2))
                )
            )
        )
    );
    return ret;
}

/// @brief Compute the cofactor of a 4x4 matrix.
pure_fn mat4 cofactor_mat4(const mat4 a) {
    // determinants (the ids of the determinants (AB - AB)):
    // 2 -> 0 2 - 2 0
    // 3 -> 0 3 - 2 1
    // 4 -> 1 2 - 3 0
    // 5 -> 1 3 - 3 1
    // 1 -> a0a3 - a1a2
    // 6 -> b0b3 - b1b2
    
    // For cofactor of Block A:
    //  Dets    blocks
    // + - +    b b a
    // 4 5 6    3 2 3
    // 3 2 -6   2 3 2
    // 5 4 -6   0 1 1
    // 2 3 6    1 0 0
    // A B C
    
    // For cofactor of Block B:
    //  Dets    blocks
    // + - +    a a b
    // 5 3 1    2 3 3
    // 2 4 -1   3 2 2
    // 3 5 -1   1 0 1
    // 4 2 1    0 1 0


    mat4 ret;
    __m256 dets_16 = _mm256_mul_ps(
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(1, 3, 1, 3))
    );
    dets_16 = _mm256_permute2f128_ps(
        _mm256_hsub_ps(
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(1, 0, 0, 1))
        ),
        dets_16,
        0b00000001
    );

    __m256 dets_A = _mm256_permute2f128_ps(
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 0, 1)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 1, 0)),
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 3, 2, 3))
            )
        ),
        a.b0,
        0b00000001
    );
    ret.b0 = _mm256_mul_ps(a.b0, dets_A);

    ret.b1 = _mm256_fms_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 2, 3)),
        _mm256_permute_mac(dets_16, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm256_permute_mac(dets_A, _MM_SHUFFLE(3, 2, 0, 1)),
            _mm256_permute_mac(ret.b0, _MM_SHUFFLE(0, 1, 3, 2))
        )
    );

    ret.b0 = _mm256_fma_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 2, 3)),
        dets_A,
        _mm256_fms_mac(
                _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 2, 3)),
                _mm256_permute_mac(dets_16, _MM_SHUFFLE(0, 1, 1, 0)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 3, 2)),
                _mm256_permute_mac(dets_A, _MM_SHUFFLE(1, 0, 3, 2))
            )
        )
    );

    return ret;
}

/// @brief Compute the adjoint of a 4x4 matrix.
pure_fn mat4 adj_mat4(const mat4 a) {
    // refer to `cofactor_mat4` for logic details
    mat4 ret;
    __m256 dets_16 = _mm256_mul_ps(
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(1, 3, 1, 3))
    );
    dets_16 = _mm256_permute2f128_ps(
        _mm256_hsub_ps(
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(1, 0, 0, 1))
        ),
        dets_16,
        0b00000001
    );

    __m256 dets_A = _mm256_permute2f128_ps(
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 0, 1)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 1, 0)),
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 3, 2, 3))
            )
        ),
        a.b0,
        0b00000001
    );
    ret.b0 = _mm256_mul_ps(a.b0, dets_A);

    ret.b1 = _mm256_fms_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 2, 1, 3)),
        _mm256_permute_mac(dets_16, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(1, 2, 0, 3)),
            _mm256_permute_mac(dets_A, _MM_SHUFFLE(3, 0, 2, 1)),
            _mm256_permute_mac(ret.b0, _MM_SHUFFLE(0, 3, 1, 2))
        )
    );
    ret.b1 = _mm256_permute2f128_ps(ret.b1, ret.b1, 0b00000001);

    ret.b0 = _mm256_fma_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(1, 2, 0, 3)),
        _mm256_permute_mac(dets_A, _MM_SHUFFLE(3, 1, 2, 0)),
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 2, 1, 3)),
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(0, 1, 1, 0)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 3, 1, 2)),
                _mm256_permute_mac(dets_A, _MM_SHUFFLE(1, 3, 0, 2))
            )
        )
    );

    return ret;
}

/// @brief Compute the inverse of a 4x4 matrix. 
pure_fn mat4 inv_mat4(const mat4 a) {
    // refer to `cofactor_mat4` for logic details
    mat4 ret;
    __m256 dets_16 = _mm256_mul_ps(
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(2, 0, 2, 0)),
        _mm256_shuffle_ps(a.b1, a.b0, _MM_SHUFFLE(1, 3, 1, 3))
    );
    dets_16 = _mm256_permute2f128_ps(
        _mm256_hsub_ps(
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(1, 0, 0, 1))
        ),
        dets_16,
        0b00000001
    );

    __m256 dets_A = _mm256_permute2f128_ps(
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 0, 1)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 1, 0)),
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 3, 2, 3))
            )
        ),
        a.b0,
        0b00000001
    );

    ret.b0 = _mm256_fma_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(1, 2, 0, 3)),
        _mm256_permute_mac(dets_A, _MM_SHUFFLE(3, 1, 2, 0)),
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(0, 2, 1, 3)),
            _mm256_permute_mac(dets_16, _MM_SHUFFLE(0, 1, 1, 0)),
            _mm256_mul_ps(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 3, 1, 2)),
                _mm256_permute_mac(dets_A, _MM_SHUFFLE(1, 3, 0, 2))
            )
        )
    );
    __m256 tmp = _mm256_mul_ps(a.b0, dets_A);

    ret.b1 = _mm256_fms_mac(
        _mm256_permute_mac(a.b1, _MM_SHUFFLE(0, 2, 1, 3)),
        _mm256_permute_mac(dets_16, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm256_fms_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(1, 2, 0, 3)),
            _mm256_permute_mac(dets_A, _MM_SHUFFLE(3, 0, 2, 1)),
            _mm256_permute_mac(tmp, _MM_SHUFFLE(0, 3, 1, 2))
        )
    );
    __m256 inv_det = _mm256_rcp_mac(_mm256_dp_ps(
        _mm256_shuffle_ps(a.b0, a.b1, _MM_SHUFFLE(1, 0, 1, 0)),
        _mm256_shuffle_ps(ret.b0, ret.b1, _MM_SHUFFLE(2, 0, 2, 0)),
        0b11111111
    ));
    ret.b1 = _mm256_permute2f128_ps(ret.b1, ret.b1, 0b00000001);
    
    ret.b0 = _mm256_mul_ps(ret.b0, inv_det);
    ret.b1 = _mm256_mul_ps(ret.b1, inv_det);

    return ret;
}

/// @brief Compute the square of a 4x4 matrix.
pure_fn mat4 sqr_mat4(const mat4 a) {
    mat4 ret;
    ret.b1 = _mm256_permute2f128_ps(a.b1, a.b1, 0b00000001); // use as temporary variable
    ret.b0 = _mm256_fma_mac(
        _mm256_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0)), a.b0,
        _mm256_fma_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1)),
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_fma_mac(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0)), ret.b1,
                _mm256_mul_ps(
                    _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1)),
                    _mm256_permute_mac(ret.b1, _MM_SHUFFLE(1, 0, 3, 2))
                )
        )
        )
    );
    ret.b1 = _mm256_permute2f128_ps(a.b0, a.b0, 0b00000001); // use as temporary variable
    ret.b1 = _mm256_fma_mac(
        _mm256_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0)), a.b1,
        _mm256_fma_mac(
            _mm256_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1)),
            _mm256_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_fma_mac(
                _mm256_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0)), ret.b1,
                _mm256_mul_ps(
                    _mm256_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1)),
                    _mm256_permute_mac(ret.b1, _MM_SHUFFLE(1, 0, 3, 2))
                )
            )
        )
    );
    return ret;
}

/// @brief Compute the Nth positive integral power of a 4x4 matrix.
pure_fn mat4 pow_mat4(const mat4 a, const unsigned N) {
    mat4 res = create_mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
    mat4 base = a;
    unsigned n = N;
    
    while (n > 0) {
        if (n % 2 == 1) {
            res = mul_mat4(res, base);
        }
        base = sqr_mat4(base);
        n /= 2;
    }
    
    return res;
}








/// @brief Store a 4x4 block matrix as an array.
void store_mat4(float *arr, const mat4 a) {
    _mm_store_ps(arr,
        _mm_movelh_ps(
            _mm256_castps256_ps128(a.b0),
            _mm256_castps256_ps128(a.b1)
        )
    );
    _mm_store_ps(arr + 4,
        _mm_movehl_ps(
            _mm256_castps256_ps128(a.b1),
            _mm256_castps256_ps128(a.b0)
        )
    );
    _mm_store_ps(arr + 8,
        _mm_movelh_ps(
            _mm256_extractf128_ps(a.b1, 1),
            _mm256_extractf128_ps(a.b0, 1)
        )
    );
    _mm_store_ps(arr + 12,
        _mm_movehl_ps(
            _mm256_extractf128_ps(a.b0, 1),
            _mm256_extractf128_ps(a.b1, 1)
        )
    );
}
/// @brief Print a 4x4 block matrix.
void print_mat4(const mat4 a) {
    _Alignas(16) float arr[16];
    memcpy(arr, &a, sizeof(mat4));
    // 0  1  8  9
    // 2  3  10 11
    // 12 13 4  5
    // 14 15 6  7

    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        arr[0], arr[1], arr[8], arr[9],
        arr[2], arr[3], arr[10], arr[11],
        arr[12], arr[13], arr[4], arr[5],
        arr[14], arr[15], arr[6], arr[7]
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
