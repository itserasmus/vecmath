/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat3` related functions for the AVX512 implementation of VecMath.
 */
#ifndef MAT3X3_H_AVX512
#define MAT3X3_H_AVX512
#include "vec_math_avx512.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
#endif
extern "C" {
#endif


/// @brief Adds two 3x3 matrices.
pure_fn mat3 add_mat3(const mat3 a, const mat3 b) {
    return _mm512_add_ps(a, b);
}

/// @brief Subtracts two 3x3 matrices.
pure_fn mat3 sub_mat3(const mat3 a, const mat3 b) {
    return _mm512_add_ps(a, b);
}

/// @brief Multiplies a 3x3 matrix by a scalar.
pure_fn mat3 scal_mul_mat3(const mat3 a, const float b) {
    return _mm512_mul_ps(a, _mm512_set1_ps(b));
}

/// @brief Transposes a 3x3 matrix.
pure_fn mat3 trans_mat3(const mat3 a) {
    return _mm512_permutexvar_ps(_mm512_setr_epi32(
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15
    ), a);
}

/// @brief Computes the determinant of a 3x3 matrix.
pure_fn float det_mat3(const mat3 a) {
    // det = 
    // c0 * (b1a2 - b2a1) +
    // c1 * (b2a0 - b0a2) +
    // c2 * (b0a1 - b1a0)
    // m0 * (C1C0  -C0C1)
    __m256 c0c1 = _mm256_mul_ps(
        _mm256_permute_mac(_mm512_castps512_ps256(a), _MM_SHUFFLE(0, 1, 0, 2)),
        _mm256_permute_mac(
            _mm256_permute2f128_ps(_mm512_castps512_ps256(a), _mm512_castps512_ps256(a), 0b01),
            _MM_SHUFFLE(0, 0, 2, 1)
        )
    );
    __m128 diff = _mm_sub_ps(
        _mm256_extractf128_ps(c0c1, 1),
        _mm256_castps256_ps128(c0c1)
    );
    return _mm_cvtss_f32(_mm_dp_ps(_mm512_extractf32x4_ps(a, 2), diff, 0b01110001));
}

/// @brief Multiplies a 3-vector by a 3x3 matrix.
pure_fn vec3 mul_vec3_mat3(const vec3 a, const mat3 b) {
    __m512 prod = _mm512_mul_ps(_mm512_permutexvar_ps(
        _mm512_setr_epi32(0,0,0,0,1,1,1,1,2,2,2,2,/*junk*/3,3,3,3), _mm512_castps128_ps512(a)), b);
    return _mm_add_ps(
        _mm512_castps512_ps128(prod),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 1),
        _mm512_extractf32x4_ps(prod, 2)
    ));
}

/// @brief Multiplies a 3x3 matrix by a 3-vector.
pure_fn vec3 mul_mat3_vec3(const mat3 a, const vec3 b) {
    __m512 prod = _mm512_permutexvar_ps(
        _mm512_setr_epi32(0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15),
        // transpose cuz AVX512 is soooo STUPID and you can't do a
        // permute (0,4,8,12) after taking a dp! I hate AVX512
        _mm512_mul_ps(_mm512_broadcast_f32x4(b), a));
    return _mm_add_ps(
        _mm512_castps512_ps128(prod),
        _mm_add_ps(
        _mm512_extractf32x4_ps(prod, 1),
        _mm512_extractf32x4_ps(prod, 2)
    ));
}

/// @brief Multiplies two 3x3 matrices.
pure_fn mat3 mul_mat3(const mat3 a, const mat3 b) {
    // To anyone criticizing this, yes, I know it's possible to
    // do it in just two multiplication operations, but that
    // involves more permutations, and multiplication operations
    // have a lower throughput than permutations. I hate AVX512.
    // AVX is so much more workable with, and has actually
    // *useful* features. Who even made this instruction set
    // like they removed half the fast operations and replaced
    // them with useless operations like that's just dumb
    return _mm512_fma_mac(
        _mm512_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0)),
        // Can't use _mm512_permute4f128 cuz it got removed
        // for no good reason. Thanks, Intel, for making me
        // use a high latency function.
        _mm512_permutexvar_ps(_mm512_setr4_epi32(0, 1, 2, 3), b),
        _mm512_fma_mac(
            _mm512_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm512_permutexvar_ps(_mm512_setr4_epi32(4, 5, 6, 7), b),
            _mm512_mul_ps(
                _mm512_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm512_permutexvar_ps(_mm512_setr4_epi32(8, 9, 10, 11), b)
            )
        )
    );
    // Despite Intel's best efforts to ruin AVX512, This function
    // is sufficiently fast to compute matrix multiplication
    // before the heat death of the universe (usually).
}

/// @brief Computes the cofator of a 3x3 matrix.
pure_fn mat3 cofactor_mat3(const mat3 a) {
    __m512 perm1 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        5, 6, 4, 0,
        2, 0, 1, 0,
        1, 2, 0, 0, /*junk values (Just like this instruction set)*/ 0, 0, 0, 0
    ), a);
    __m512 perm2 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        10, 8, 9, 0,
        9, 10, 8, 0,
        6, 4, 5, 0, /*junk values (At least it combines permute and permute4f128)*/ 0, 0, 0, 0
    ), a);
    return _mm512_sub_ps(
        _mm512_mul_ps(perm1, perm2),
        _mm512_mul_ps(
            _mm512_permute_mac(perm1, _MM_SHUFFLE(3, 1, 0, 2)),
            _mm512_permute_mac(perm2, _MM_SHUFFLE(3, 0, 2, 1))
        )
    ); 
}

/// @brief Computes the adjoint of a 3x3 matrix.
pure_fn mat3 adj_mat3(const mat3 a) {
    __m512 perm1 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        5, 9, 1, 0,
        6, 10, 2, 0,
        4, 8, 0, 0, /*junk values (just like my decision to add AVX512. could’ve spent
            that time making something that offers actual performance benefits)*/ 0, 0, 0, 0
    ), a);
    __m512 perm2 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        10, 2, 6, 0,
        8, 0, 4, 0,
        9, 1, 5, 0, /*junk values (because what else would you expect from AVX512?)*/ 0, 0, 0, 0
    ), a);
    return _mm512_sub_ps(
        _mm512_mul_ps(perm1, perm2),
        _mm512_mul_ps(
            _mm512_permute_ps(perm1, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm512_permute_ps(perm2, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    // "Premature optimization is the root of all evil."
    // Wrong. It's AVX512.
}

/// @brief Computes the inverse of a 3x3 matrix.
pure_fn mat3 inv_mat3(const mat3 a) {
    __m512 perm1 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        5, 9, 1, 0,
        6, 10, 2, 0,
        4, 8, 0, 0, /*junk values*/ 0, 0, 0, 0
    ), a);
    __m512 perm2 = _mm512_permutexvar_ps(_mm512_setr_epi32(
        10, 2, 6, 0,
        8, 0, 4, 0,
        9, 1, 5, 0, /*junk values (just like the logic behind AVX512: less speed,
            more headaches, and way more registers)*/ 0, 0, 0, 0
    ), a);
    __m512 adj = _mm512_sub_ps(
        _mm512_mul_ps(perm1, perm2),
        _mm512_mul_ps(
            _mm512_permute_mac(perm1, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm512_permute_mac(perm2, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    __m128 det = _mm_rcp_mac(_mm_dp_ps(
        // I can't believe I have to switch back to SSE for this.
        _mm512_castps512_ps128(_mm512_permutexvar_ps(_mm512_setr_epi32(
            0, 4, 8, 12, /*junk values*/0,0,0,0,0,0,0,0,0,0,0,0
        ), adj)),
        _mm512_castps512_ps128(a),
        0b01110111
    ));
    return _mm512_mul_ps(
        _mm512_broadcast_f32x4(det),
        adj
    );
    // "Perfection is not when there is no more to add, but no
    // more to take away."
    // Except for AVX512, where it's all about adding unnecessary
    // complexity and performance hits.
}

/// @brief Computes the square of a 3x3 matrix.
pure_fn mat3 sqr_mat3(const mat3 a) {
    return _mm512_fma_mac(
        _mm512_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0)),
        // Can't use _mm512_permute4f128 cuz it got removed
        // for no good reason. Thanks, Intel, for making me
        // use a high latency function.
        _mm512_permutexvar_ps(_mm512_setr4_epi32(0, 1, 2, 3), a),
        _mm512_fma_mac(
            _mm512_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm512_permutexvar_ps(_mm512_setr4_epi32(4, 5, 6, 7), a),
            _mm512_mul_ps(
                _mm512_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2)),
                _mm512_permutexvar_ps(_mm512_setr4_epi32(8, 9, 10, 11), a)
            )
        )
    );
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
void store_mat3(float *arr, const mat3 a) { // arr must be at least 16 wide
    _mm512_storeu_ps(arr, a);
}

/// @brief Prints a 3x3 matrix.
void print_mat3(const mat3 a) {
    _Alignas(16) float a_arr[16];
    _mm512_store_ps(a_arr, a);
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
