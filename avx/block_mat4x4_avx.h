#ifndef BLOCK_MAT4X4_H_AVX
#define BLOCK_MAT4X4_H_AVX
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

// pure_fn vec4 mul_vec4_mat4(const vec4 a, const mat4 b)

// pure_fn vec4 mul_mat4_vec4(const mat4 a, const vec4 b)

// pure_fn mat4 mul_mat4(const mat4 a, const mat4 b)

// pure_fn mat4 cofactor_mat4(const mat4 a)

// pure_fn mat4 adj_mat4(const mat4 a)

// pure_fn mat4 inv_mat4(const mat4 a)

// pure_fn mat4 sqr_mat4(const mat4 a)

// pure_fn mat4 pow_mat4(const mat4 a, const unsigned N)








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
