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


/// @brief Add two 4x4 matrices.
pure_fn rmat4 add_rmat4(const rmat4 a, const rmat4 b) {
    rmat4 ret;
    ret.m0 = _mm256_add_ps(a.m0, b.m0);
    ret.m1 = _mm256_add_ps(a.m1, b.m1);
    return ret;
}

/// @brief Subtract two 4x4 matrices.
pure_fn rmat4 sub_rmat4(const rmat4 a, const rmat4 b) {
    rmat4 ret;
    ret.m0 = _mm256_sub_ps(a.m0, b.m0);
    ret.m1 = _mm256_sub_ps(a.m1, b.m1);
    return ret;
}

/// @brief Multtiply a 4x4 matrix by a scalar.
pure_fn rmat4 scal_mul_rmat4(const rmat4 a, const float b) {
    rmat4 ret;
    __m256 b_vec = _mm256_set1_ps(b);
    ret.m0 = _mm256_mul_ps(a.m0, b_vec);
    ret.m1 = _mm256_mul_ps(a.m1, b_vec);
    return ret;
}

/// @brief Transpose a 4x4 matrix.
pure_fn rmat4 trans_rmat4(const rmat4 a) {
    rmat4 ret;
    ret.m0 = _mm256_permute2f128_ps(a.m0, a.m1, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(a.m0, a.m1, 0b00110001);
    __m256 tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 2, 3, 2));
    ret.m0 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001);
    tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(2, 0, 2, 0));
    tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 1, 3, 1));
    ret.m0 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001);
    return ret;
}

/// @brief Compute the determinant of a 4x4 matrix.
pure_fn float det_rmat4(const rmat4 a) {
    // det = 
    // (a0b0 - a0b0) * (a1b1 - a1b1)
    //   0 0    0 0      0 0    0 0
    //   0 1    1 0      3 2    2 3
    //   0 2    2 0      1 3    3 1
    //   0 3    3 0      2 1    1 2
    // (a1b1 - a1b1) * (a0b0 - a0b0)
    //   0 0    0 0      0 0    0 0
    //   0 1    1 0      3 2    2 3
    //   0 2    2 0      1 3    3 1
    //   0 3    3 0      2 1    1 2
    //   c0     c1       c2     c3
    __m256 lane_swap0 = _mm256_permute2f128_ps(a.m0, a.m0, 0b00000001);
    __m256 lane_swap1 = _mm256_permute2f128_ps(a.m1, a.m1, 0b00000001);
    __m256 dp = _mm256_dp_ps(
        _mm256_fms_mac(
            _mm256_permute_mac(lane_swap0, _MM_SHUFFLE(2, 1, 3, 0)),
            _mm256_permute_mac(lane_swap1, _MM_SHUFFLE(1, 3, 2, 0)),
            _mm256_mul_ps(
                _mm256_permute_mac(lane_swap0, _MM_SHUFFLE(1, 3, 2, 0)),
                _mm256_permute_mac(lane_swap1, _MM_SHUFFLE(2, 1, 3, 0))
            )
        ),
        _mm256_fms_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)), a.m1,
            _mm256_mul_ps(a.m0, _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)))
        ),
        0b11101111
    );
    return _mm_cvtss_f32(_mm256_castps256_ps128(dp)) + _mm_cvtss_f32(_mm256_extractf128_ps(dp, 1));
}

/// @brief Multiply a 4-vector by a 4x4 matrix.
pure_fn vec4 mul_vec4_rmat4(const vec4 a, const rmat4 b) {
    __m256 tmp0 = _mm256_fma_mac(
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0))
        ),
        b.m0, 
        _mm256_mul_ps(
            _mm256_set_m128(
                _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 3, 3)),
                _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2))
            ),
            b.m1
        )
    );
    return _mm_add_ps(
        _mm256_castps256_ps128(tmp0),
        _mm256_extractf128_ps(tmp0, 1)
    );
}

/// @brief Multiply a 4x4 matrix by a 4-vector.
pure_fn vec4 mul_rmat4_vec4(const rmat4 a, const vec4 b) {
    __m256 b_exp = _mm256_set_m128(b, b); // expanded b
    __m256 tmp0 = _mm256_blend_ps(
        _mm256_dp_ps(a.m0, b_exp, 0b11110001),
        _mm256_dp_ps(a.m1, b_exp, 0b11110010),
        0b00100010
    );
    vec4 ret = _mm_movelh_ps(
        _mm256_castps256_ps128(tmp0),
        _mm256_extractf128_ps(tmp0, 1)
    );
    return _mm_permute_mac(ret, _MM_SHUFFLE(3, 1, 2, 0));
}

/// @brief Multiply two 4x4 matrices.
pure_fn rmat4 mul_rmat4(const rmat4 a, const rmat4 b) {
    rmat4 ret;
    ret.m0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)), _mm256_permute2f128_ps(b.m0, b.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 1, 1, 1)), _mm256_permute2f128_ps(b.m0, b.m0, 0b00010001),
            _mm256_fma_mac(
                _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 2, 2, 2)), _mm256_permute2f128_ps(b.m1, b.m1, 0b00000000),
                _mm256_mul_ps(_mm256_permute_mac(a.m0, _MM_SHUFFLE(3, 3, 3, 3)), _mm256_permute2f128_ps(b.m1, b.m1, 0b00010001))
            )
        )
    );
    ret.m1 = _mm256_fma_mac(
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)), _mm256_permute2f128_ps(b.m0, b.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 1, 1, 1)), _mm256_permute2f128_ps(b.m0, b.m0, 0b00010001),
            _mm256_fma_mac(
                _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 2, 2, 2)), _mm256_permute2f128_ps(b.m1, b.m1, 0b00000000),
                _mm256_mul_ps(_mm256_permute_mac(a.m1, _MM_SHUFFLE(3, 3, 3, 3)), _mm256_permute2f128_ps(b.m1, b.m1, 0b00010001))
            )
        )
    );
    return ret;
}

/// @brief Compute the cofactor of a 4x4 matrix.
pure_fn rmat4 cofactor_rmat4(const rmat4 a) {
    rmat4 ret;
    // determinants (the columns of the determinant):
    // 1 -> 0 1 - 1 0
    // 2 -> 0 2 - 2 0
    // 3 -> 0 3 - 3 0
    // 4 -> 1 2 - 2 1
    // 5 -> 1 3 - 3 1
    // 6 -> 2 3 - 3 2   
    // 
    // first 3 columns are the adjacent row "id" and
    // last three are the determinants to mulitply them with
    // (b b b) * (D D  D)
    //  2 3 1     5 4 -6
    //  3 2 0     2 3 +6
    //  1 0 3     3 5 -1
    //  0 1 2     4 2 +1

    // calculate -6 +6 -1 +1
    __m256 dets = _mm256_permute2f128_ps(_mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 2, 3)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 2, 3)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 0, 1))
        )
    ), a.m0, 0b00000001);
    __m256 dets6611 =_mm256_permute2f128_ps( _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 2, 3)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 3, 2)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 2, 3))
        )
    ), a.m0, 0b00000001);
    ret.m0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 3, 2)), dets,
        _mm256_fms_mac(
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 0, 1)), dets6611,
            _mm256_mul_ps(
                _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 3)),
                _mm256_permute_mac(dets, _MM_SHUFFLE(1, 0, 2, 3))
            )
        )
    );
    ret.m1 = _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_permute_mac(dets, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 3, 2)), dets,
            _mm256_mul_ps(_mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 0, 1)), dets6611)
        )
    );
    return ret;
}

/// @brief Compute the adjoint of a 4x4 matrix.
pure_fn rmat4 adj_rmat4(const rmat4 a) {
    rmat4 ret;
    ret.m0 = _mm256_permute2f128_ps(_mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 2, 3)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 2, 3)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 0, 1))
        )
    ), a.m0, 0b00000001);
    ret.m1 =_mm256_permute2f128_ps( _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 2, 3)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 3, 2)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 2, 3))
        )
    ), a.m0, 0b00000001);
    __m256 tmp0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 3, 2)), ret.m0,
        _mm256_fms_mac(
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 0, 1)), ret.m1,
            _mm256_mul_ps(
                _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 3)),
                _mm256_permute_mac(ret.m0, _MM_SHUFFLE(1, 0, 2, 3))
            )
        )
    );
    __m256 tmp1 = _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_permute_mac(ret.m0, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 3, 2)), ret.m0,
            _mm256_mul_ps(_mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 0, 1)), ret.m1)
        )
    );
    ret.m0 = _mm256_permute2f128_ps(tmp0, tmp1, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp0, tmp1, 0b00110001);
    __m256 tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 2, 3, 2));
    ret.m0 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001);
    tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(2, 0, 2, 0));
    tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 1, 3, 1));
    ret.m0 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001);
    return ret;
}

/// @brief Compute the inverse of a 4x4 matrix.
pure_fn rmat4 inv_rmat4(const rmat4 a) {
    rmat4 ret;
    ret.m0 = _mm256_permute2f128_ps(_mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 2, 3)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 2, 3)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 0, 1))
        )
    ), a.m0, 0b00000001);
    ret.m1 =_mm256_permute2f128_ps( _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 2, 3)),
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 3, 2)),
        _mm256_mul_ps(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 2, 3))
        )
    ), a.m0, 0b00000001);
    __m256 tmp0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 3, 2)), ret.m0,
        _mm256_fms_mac(
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 0, 1)), ret.m1,
            _mm256_mul_ps(
                _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 3)),
                _mm256_permute_mac(ret.m0, _MM_SHUFFLE(1, 0, 2, 3))
            )
        )
    );
    __m256 tmp1 = _mm256_fms_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_permute_mac(ret.m0, _MM_SHUFFLE(1, 0, 2, 3)),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 3, 2)), ret.m0,
            _mm256_mul_ps(_mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 0, 1)), ret.m1)
        )
    );
    __m256 inv_det = _mm256_rcp_mac(_mm256_dp_ps(a.m0, tmp0, 0b11111111));
    ret.m0 = _mm256_permute2f128_ps(tmp0, tmp1, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp0, tmp1, 0b00110001);
    __m256 tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 2, 3, 2));
    ret.m0 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000);
    ret.m1 = _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001);
    tmp2 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(2, 0, 2, 0));
    tmp3 = _mm256_shuffle_ps(ret.m0, ret.m1, _MM_SHUFFLE(3, 1, 3, 1));
    ret.m0 = _mm256_mul_ps(inv_det, _mm256_permute2f128_ps(tmp2, tmp3, 0b00100000));
    ret.m1 = _mm256_mul_ps(inv_det, _mm256_permute2f128_ps(tmp2, tmp3, 0b00110001));
    return ret;
}

/// @brief Compute the square of a 4x4 matrix.
pure_fn rmat4 sqr_rmat4(const rmat4 a) {
    rmat4 ret;
    ret.m0 = _mm256_fma_mac(
        _mm256_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)), _mm256_permute2f128_ps(a.m0, a.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m0, _MM_SHUFFLE(1, 1, 1, 1)), _mm256_permute2f128_ps(a.m0, a.m0, 0b00010001),
            _mm256_fma_mac(
                _mm256_permute_mac(a.m0, _MM_SHUFFLE(2, 2, 2, 2)), _mm256_permute2f128_ps(a.m1, a.m1, 0b00000000),
                _mm256_mul_ps(_mm256_permute_mac(a.m0, _MM_SHUFFLE(3, 3, 3, 3)), _mm256_permute2f128_ps(a.m1, a.m1, 0b00010001))
            )
        )
    );
    ret.m1 = _mm256_fma_mac(
        _mm256_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)), _mm256_permute2f128_ps(a.m0, a.m0, 0b00000000),
        _mm256_fma_mac(
            _mm256_permute_mac(a.m1, _MM_SHUFFLE(1, 1, 1, 1)), _mm256_permute2f128_ps(a.m0, a.m0, 0b00010001),
            _mm256_fma_mac(
                _mm256_permute_mac(a.m1, _MM_SHUFFLE(2, 2, 2, 2)), _mm256_permute2f128_ps(a.m1, a.m1, 0b00000000),
                _mm256_mul_ps(_mm256_permute_mac(a.m1, _MM_SHUFFLE(3, 3, 3, 3)), _mm256_permute2f128_ps(a.m1, a.m1, 0b00010001))
            )
        )
    );
    return ret;
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
    _mm256_storeu_ps(arr, a.m0);
    _mm256_storeu_ps(arr + 8, a.m1);
}

/// @brief Print a 4x4 matrix.
void print_rmat4(const rmat4 a) {
    _Alignas(16) float arr[16];
    _mm256_store_ps(arr, a.m0);
    _mm256_store_ps(arr + 8, a.m1);
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
