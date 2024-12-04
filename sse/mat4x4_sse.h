/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains all `mat4` related functions for the SSE implementation of VecMath.
 */
#ifndef MAT4X4_H_SSE
#define MAT4X4_H_SSE
#include "vec_math_sse.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif

/// @brief Add two 4x4 matrices.
pure_fn mat4 add_mat4(const mat4 a, const mat4 b) {
    mat4 ret;
    ret.b0 = _mm_add_ps(a.b0, b.b0);
    ret.b1 = _mm_add_ps(a.b1, b.b1);
    ret.b2 = _mm_add_ps(a.b2, b.b2);
    ret.b3 = _mm_add_ps(a.b3, b.b3);
    return ret;
}

/// @brief Subtract two 4x4 matrices.
pure_fn mat4 sub_mat4(const mat4 a, const mat4 b) {
    mat4 ret;
    ret.b0 = _mm_sub_ps(a.b0, b.b0);
    ret.b1 = _mm_sub_ps(a.b1, b.b1);
    ret.b2 = _mm_sub_ps(a.b2, b.b2);
    ret.b3 = _mm_sub_ps(a.b3, b.b3);
    return ret;
}

/// @brief Multiply a 4x4 matrix by a scalar.
pure_fn mat4 scal_mul_mat4(const mat4 a, const float b) {
    __m128 b_vec = _mm_set_ps1(b);
    mat4 ret;
    ret.b0 = _mm_mul_ps(a.b0, b_vec);
    ret.b1 = _mm_mul_ps(a.b1, b_vec);
    ret.b2 = _mm_mul_ps(a.b2, b_vec);
    ret.b3 = _mm_mul_ps(a.b3, b_vec);
    return ret;
}

/// @brief Transpose a 4x4 matrix.
pure_fn mat4 trans_mat4(const mat4 a) {
    mat4 ret;
    ret.b0 = _mm_permute_mac(a.b0, _MM_SHUFFLE(3, 1, 2, 0));
    ret.b1 = _mm_permute_mac(a.b2, _MM_SHUFFLE(3, 1, 2, 0));
    ret.b2 = _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 1, 2, 0));
    ret.b3 = _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 1, 2, 0));
    return ret;
}

/// @brief Multiply a 4-vector by a 4x4 matrix.
pure_fn vec4 mul_vec4_mat4(const vec4 a, const mat4 b) {
    __m128 a_0011 = _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 a_2233 = _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 2, 2));
    __m128 tmp1 = _mm_fma_mac(
        a_0011, b.b0,
        _mm_mul_ps(a_2233, b.b2)
    );
    __m128 tmp2 = _mm_fma_mac(
        a_0011, b.b1,
        _mm_mul_ps(a_2233, b.b3)
    );
    return _mm_hadd_ps(
        _mm_permute_mac(tmp1, _MM_SHUFFLE(3, 1, 2, 0)),
        _mm_permute_mac(tmp2, _MM_SHUFFLE(3, 1, 2, 0))
    );
}

/// @brief Multiply a 4x4 matrix by a 4-vector.
pure_fn vec4 mul_mat4_vec4(const mat4 a, const vec4 b) {
    __m128 b0101 = _mm_movelh_ps(b, b);
    __m128 b2323 = _mm_movehl_ps(b, b);
    
    return _mm_hadd_ps(
        _mm_fma_mac(
            b0101, a.b0,
            _mm_mul_ps(b2323, a.b1)
        ),
        _mm_fma_mac(
            b0101, a.b2,
            _mm_mul_ps(b2323, a.b3)
        )
    );
}

/// @brief Multiply two 4x4 matrices.
pure_fn mat4 mul_mat4(const mat4 a, const mat4 b) {
    __m128 aperm3 = _mm_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1));
    __m128 aperm2 = _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0));
    __m128 aperm1 = _mm_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1));
    __m128 aperm0 = _mm_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0));
    
    mat4 ret;
    ret.b0 = _mm_fma_mac(
        aperm0, b.b0,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(b.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, b.b2,
                _mm_mul_ps(aperm3, _mm_permute_mac(b.b2, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    ret.b1 = _mm_fma_mac(
        aperm0, b.b1,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(b.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, b.b3,
                _mm_mul_ps(aperm3, _mm_permute_mac(b.b3, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    aperm3 = _mm_permute_mac(a.b3, _MM_SHUFFLE(2, 2, 1, 1));
    aperm2 = _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 3, 0, 0));
    aperm1 = _mm_permute_mac(a.b2, _MM_SHUFFLE(2, 2, 1, 1));
    aperm0 = _mm_permute_mac(a.b2, _MM_SHUFFLE(3, 3, 0, 0));
    
    ret.b2 = _mm_fma_mac(
        aperm0, b.b0,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(b.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, b.b2,
                _mm_mul_ps(aperm3, _mm_permute_mac(b.b2, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    ret.b3 = _mm_fma_mac(
        aperm0, b.b1,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(b.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, b.b3,
                _mm_mul_ps(aperm3, _mm_permute_mac(b.b3, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );

    return ret;
}

/// @brief Compute the determinant of a 4x4 matrix.
pure_fn float det_mat4(const mat4 a) {
    // det =
    // (a0a3 - a1a2)(c0c3 - c1c2) +
    // (b0b3 - b1b2)(d0d3 - d1d2) +
    // ( A0  -  B0 )( A1  -  B1 )
    // (a0b3 - a2b1)(c1d2 - c3d0) +
    // (a1b2 - a3b0)(c0d3 - c2d1) +
    // (a2b0 - a0b2)(c1d3 - c3d1) +
    // (a3b1 - a1b3)(c0d2 - c2d0)
    // ( C0  -  C1 )( C2  -  C3 )

    __m128 dets = _mm_hsub_ps(
        _mm_mul_ps(_mm_movelh_ps(a.b0, a.b1), _mm_shuffle_ps(a.b0, a.b1, _MM_SHUFFLE(2, 3, 2, 3))),
        _mm_mul_ps(_mm_movelh_ps(a.b2, a.b3), _mm_shuffle_ps(a.b2, a.b3, _MM_SHUFFLE(2, 3, 2, 3)))
    );
    printf("%f %f\n",_mm_cvtss_f32(_mm_dp_ps(dets, _mm_permute_mac(dets, _MM_SHUFFLE(0, 1, 2, 3)), 0b00110001)),_mm_cvtss_f32(_mm_dp_ps(
        _mm_fms_mac(
            a.b0, _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 3, 2)), _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 2, 0, 1)))
        ),
        _mm_fms_mac(
            _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 1, 0, 1)), _mm_permute_mac(a.b3, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(2, 3, 2, 3)), _mm_permute_mac(a.b3, _MM_SHUFFLE(0, 1, 1, 0)))
        ),
        0b11110001
    )));
    return _mm_cvtss_f32(_mm_dp_ps(dets, _mm_permute_mac(dets, _MM_SHUFFLE(0, 1, 2, 3)), 0b00110001)) +
        _mm_cvtss_f32(_mm_dp_ps(
        _mm_fms_mac(
            a.b0, _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 2, 3)),
            _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 3, 2)), _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 2, 0, 1)))
        ),
        _mm_fms_mac(
            _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 1, 0, 1)), _mm_permute_mac(a.b3, _MM_SHUFFLE(2, 3, 3, 2)),
            _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(2, 3, 2, 3)), _mm_permute_mac(a.b3, _MM_SHUFFLE(0, 1, 1, 0)))
        ),
        0b11110001
    ));
}

/// @brief Compute the cofactor of a 4x4 matrix.
pure_fn mat4 cofactor_mat4(const mat4 a) {
    mat4 ret;
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
    // 5 3 1    2 3 2
    // 2 4 -1   3 2 3
    // 3 5 -1   1 0 0
    // 4 2 1    0 1 1


    __m128 XOR_MASK = _mm_set_ps(0.0f, -0.0f, -0.0f, 0.0f);
    __m128 dets = _mm_hsub_ps(
        _mm_mul_ps(_mm_movelh_ps(a.b2, a.b3), _mm_shuffle_ps(a.b2, a.b3, _MM_SHUFFLE(2, 3, 2, 3))),
        _mm_mul_ps(_mm_movelh_ps(a.b0, a.b1), _mm_shuffle_ps(a.b0, a.b1, _MM_SHUFFLE(2, 3, 2, 3)))
    );

    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 1, 0, 1)), _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(2, 3, 2, 3)), _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b0 = _mm_fma_mac(
        _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 2, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 0, 3, 2)),
        _mm_fms_mac(
            _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 2, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(1, 1, 1, 1)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 3, 2)), dets_B)
        )
    );
    ret.b1 = _mm_fma_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 3, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 1, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b1, _MM_SHUFFLE(0, 1, 2, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(0, 0, 0, 0)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 2, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 0, 2, 3)))
        )
    );

    // FOR BLOCKS C & D

    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 1, 0, 1)), _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(2, 3, 2, 3)), _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b2 = _mm_fma_mac(
        _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 2, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 0, 3, 2)),
        _mm_fms_mac(
            _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 1, 2, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(3, 3, 3, 3)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b3, _MM_SHUFFLE(0, 1, 3, 2)), dets_B)
        )
    );
    ret.b3 = _mm_fma_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 1, 3, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 1, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b3, _MM_SHUFFLE(0, 1, 2, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(2, 2, 2, 2)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(1, 0, 2, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 0, 2, 3)))
        )
    );

    return ret;
}

/// @brief Compute the adjoint of a 4x4 matrix.
pure_fn mat4 adj_mat4(const mat4 a) {
    // Look at cofactor_mat4 for the algorithm specifics.
    // adj_mat4 simply adds transposition within the _MM_SHUFFLEs
    // which gives the same number of operations as cofactor_mat4
    // by avoiding any special transposition step.
    mat4 ret;

    __m128 XOR_MASK = _mm_set_ps(0.0f, -0.0f, -0.0f, 0.0f);
    __m128 dets = _mm_hsub_ps(
        _mm_mul_ps(_mm_movelh_ps(a.b2, a.b3), _mm_shuffle_ps(a.b2, a.b3, _MM_SHUFFLE(2, 3, 2, 3))),
        _mm_mul_ps(_mm_movelh_ps(a.b0, a.b1), _mm_shuffle_ps(a.b0, a.b1, _MM_SHUFFLE(2, 3, 2, 3)))
    );

    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 0, 1, 1)), _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(2, 2, 3, 3)), _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b0 = _mm_fma_mac(
        _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 0, 1)),
        _mm_fms_mac(
            _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(1, 1, 1, 1)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b1, _MM_SHUFFLE(0, 3, 1, 2)), dets_B)
        )
    );
    ret.b2 = _mm_fma_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 3, 1, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 2, 3, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b1, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(0, 0, 0, 0)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 1, 0, 3)))
        )
    );

    // FOR BLOCKS C & D

    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 0, 1, 1)), _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 3, 3)), _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b1 = _mm_fma_mac(
        _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 0, 1)),
        _mm_fms_mac(
            _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(3, 3, 3, 3)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b3, _MM_SHUFFLE(0, 3, 1, 2)), dets_B)
        )
    );
    ret.b3 = _mm_fma_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 3, 1, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 2, 3, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b3, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(2, 2, 2, 2)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 1, 0, 3)))
        )
    );

    return ret;
}

/// @brief Compute the inverse of a 4x4 matrix.
pure_fn mat4 inv_mat4(const mat4 a) {
    // Look at cofactor_mat4 for the algorithm specifics.
    // inv_mat4 simply adds transposition within the _MM_SHUFFLEs
    // which avoids any special transposition step, and then
    // divides by the determinant, calculated with a dot product
    mat4 ret;

    __m128 XOR_MASK = _mm_set_ps(0.0f, -0.0f, -0.0f, 0.0f);
    __m128 dets = _mm_hsub_ps(
        _mm_mul_ps(_mm_movelh_ps(a.b2, a.b3), _mm_shuffle_ps(a.b2, a.b3, _MM_SHUFFLE(2, 3, 2, 3))),
        _mm_mul_ps(_mm_movelh_ps(a.b0, a.b1), _mm_shuffle_ps(a.b0, a.b1, _MM_SHUFFLE(2, 3, 2, 3)))
    );

    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 0, 1, 1)), _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(2, 2, 3, 3)), _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b0 = _mm_fma_mac(
        _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 0, 1)),
        _mm_fms_mac(
            _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(1, 1, 1, 1)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b1, _MM_SHUFFLE(0, 3, 1, 2)), dets_B)
        )
    );
    ret.b2 = _mm_fma_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 3, 1, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 2, 3, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b1, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(0, 0, 0, 0)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 1, 0, 3)))
        )
    );

    // FOR BLOCKS C & D

    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.b0, _MM_SHUFFLE(0, 0, 1, 1)), _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 2, 2, 3)),
        _mm_mul_ps(_mm_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 3, 3)), _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 0, 1)))
    );
    
    ret.b1 = _mm_fma_mac(
        _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 3, 0, 1)),
        _mm_fms_mac(
            _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(3, 3, 3, 3)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b3, _MM_SHUFFLE(0, 3, 1, 2)), dets_B)
        )
    );
    ret.b3 = _mm_fma_mac(
        _mm_permute_mac(a.b2, _MM_SHUFFLE(0, 3, 1, 2)), _mm_permute_mac(dets_B, _MM_SHUFFLE(1, 2, 3, 0)),
        _mm_fms_mac(
            _mm_permute_mac(a.b3, _MM_SHUFFLE(0, 2, 1, 3)), _mm_xor_ps(_mm_permute_mac(dets, _MM_SHUFFLE(2, 2, 2, 2)), XOR_MASK),
            _mm_mul_ps(_mm_permute_mac(a.b2, _MM_SHUFFLE(1, 2, 0, 3)), _mm_permute_mac(dets_B, _MM_SHUFFLE(2, 1, 0, 3)))
        )
    );

    // reuse XOR_MASK to store the inverse determinant
    XOR_MASK = _mm_rcp_mac(_mm_dp_ps(
        _mm_movelh_ps(a.b0, a.b1),
        _mm_shuffle_ps(ret.b0, ret.b2, _MM_SHUFFLE(2, 0, 2, 0)),
        0b11111111
    ));

    ret.b0 = _mm_mul_ps(ret.b0, XOR_MASK);
    ret.b1 = _mm_mul_ps(ret.b1, XOR_MASK);
    ret.b2 = _mm_mul_ps(ret.b2, XOR_MASK);
    ret.b3 = _mm_mul_ps(ret.b3, XOR_MASK);
    
    return ret;
}


/// @brief Compute the square of a 4x4 matrix.
pure_fn mat4 sqr_mat4(const mat4 a) {
    __m128 aperm3 = _mm_permute_mac(a.b1, _MM_SHUFFLE(2, 2, 1, 1));
    __m128 aperm2 = _mm_permute_mac(a.b1, _MM_SHUFFLE(3, 3, 0, 0));
    __m128 aperm1 = _mm_permute_mac(a.b0, _MM_SHUFFLE(2, 2, 1, 1));
    __m128 aperm0 = _mm_permute_mac(a.b0, _MM_SHUFFLE(3, 3, 0, 0));
    
    mat4 ret;
    ret.b0 = _mm_fma_mac(
        aperm0, a.b0,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, a.b2,
                _mm_mul_ps(aperm3, _mm_permute_mac(a.b2, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    ret.b1 = _mm_fma_mac(
        aperm0, a.b1,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, a.b3,
                _mm_mul_ps(aperm3, _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    aperm3 = _mm_permute_mac(a.b3, _MM_SHUFFLE(2, 2, 1, 1));
    aperm2 = _mm_permute_mac(a.b3, _MM_SHUFFLE(3, 3, 0, 0));
    aperm1 = _mm_permute_mac(a.b2, _MM_SHUFFLE(2, 2, 1, 1));
    aperm0 = _mm_permute_mac(a.b2, _MM_SHUFFLE(3, 3, 0, 0));
    
    ret.b2 = _mm_fma_mac(
        aperm0, a.b0,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(a.b0, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, a.b2,
                _mm_mul_ps(aperm3, _mm_permute_mac(a.b2, _MM_SHUFFLE(1, 0, 3, 2)))
            )
        )
    );
    ret.b3 = _mm_fma_mac(
        aperm0, a.b1,
        _mm_fma_mac(
            aperm1, _mm_permute_mac(a.b1, _MM_SHUFFLE(1, 0, 3, 2)),
            _mm_fma_mac(
                aperm2, a.b3,
                _mm_mul_ps(aperm3, _mm_permute_mac(a.b3, _MM_SHUFFLE(1, 0, 3, 2)))
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





#define swap_mat_tmp_6969(i, j) \
    store = arr[i]; \
    arr[i] = arr[j]; \
    arr[j] = store
/// @brief Store a 4x4 block matrix as an array.
void store_mat4(float *arr, const mat4 a) {
    memcpy(arr, &a, sizeof(mat4));
    // a0 a1 b0 b1
    // a2 a3 b2 b3
    // c0 c1 d0 d1
    // c2 c3 d2 d3
    float store;
    swap_mat_tmp_6969(2, 4);
    swap_mat_tmp_6969(3, 5);
    swap_mat_tmp_6969(10, 12);
    swap_mat_tmp_6969(11, 13);
}
#undef swap_mat_tmp_6969
/// @brief Print a 4x4 block matrix.
void print_mat4(const mat4 a) {
    _Alignas(16) float arr[16];
    memcpy(arr, &a, sizeof(mat4));

    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        arr[0], arr[1], arr[4], arr[5],
        arr[2], arr[3], arr[6], arr[7],
        arr[8], arr[9], arr[12], arr[13],
        arr[10], arr[11], arr[14], arr[15]
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
