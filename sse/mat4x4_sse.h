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

// add, sub, scal_mul, mul, det, adj, inv, trans, pre_vec_mul, post_vec_mul, powers

pure_fn rmat4 add_rmat4(const rmat4 a, const rmat4 b) {
    return (rmat4) {
        .m0 = _mm_add_ps(a.m0, b.m0),
        .m1 = _mm_add_ps(a.m1, b.m1),
        .m2 = _mm_add_ps(a.m2, b.m2),
        .m3 = _mm_add_ps(a.m3, b.m3)
    };
}

pure_fn rmat4 sub_rmat4(const rmat4 a, const rmat4 b) {
    return (rmat4) {
        .m0 = _mm_sub_ps(a.m0, b.m0),
        .m1 = _mm_sub_ps(a.m1, b.m1),
        .m2 = _mm_sub_ps(a.m2, b.m2),
        .m3 = _mm_sub_ps(a.m3, b.m3)
    };
}

pure_fn rmat4 scal_mul_rmat4(const rmat4 a, const float b) {
    __m128 b_vec = _mm_set_ps1(b);
    return (rmat4) {
        .m0 = _mm_mul_ps(a.m0, b_vec),
        .m1 = _mm_mul_ps(a.m1, b_vec),
        .m2 = _mm_mul_ps(a.m2, b_vec),
        .m3 = _mm_mul_ps(a.m3, b_vec)
    };
}

pure_fn rmat4 trans_rmat4(const rmat4 a) {
    __m128 _Tmp3, _Tmp2, _Tmp1, _Tmp0;                                
    _Tmp0   = _mm_shuffle_ps(a.m0, a.m1, 0x44);
    _Tmp2   = _mm_shuffle_ps(a.m0, a.m1, 0xEE);
    _Tmp1   = _mm_shuffle_ps(a.m2, a.m3, 0x44);
    _Tmp3   = _mm_shuffle_ps(a.m2, a.m3, 0xEE);

    return (rmat4) {                                     
        .m0 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88),
        .m1 = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD),
        .m2 = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88),
        .m3 = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD)
    };
}

pure_fn float det_rmat4(const rmat4 a) {
    // det = 
    // (a0b1 - a1b0)(c2d3 - c3d2) +
    // (a1b2 - a2b1)(c0d3 - c3d0) +
    // (a2b0 - a0b2)(c1d3 - c3d1) +
    // (a3b1 - a1b3)(c0d2 - c3d0) +
    // (a0b3 - a3b0)(c1d2 - c2d1) +
    // (a2b3 - a3b2)(c0d1 - c1d0)
    // ( C0  -  C1 )( C2  -  C3 )
    float first_4 = _mm_cvtss_f32(_mm_dp_ps(
        _mm_sub_ps(
            _mm_mul_ps(a.m0, _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1))),
            _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), a.m1)
        ),
        _mm_sub_ps(
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 3, 3, 3))),
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 1, 0, 2)))
        ),
        0b11110001
    ));
    float last_2 = _mm_cvtss_f32(_mm_dp_ps(
        _mm_sub_ps(
            _mm_mul_ps(a.m0, _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 3, 0, 3))),
            _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 3, 0, 3)), a.m1)
        ),
        _mm_sub_ps(
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 0, 1)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 1, 0, 2))),
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 0, 0, 1)))
        ),
        0b001010001
    ));
    return first_4 + last_2;
}

pure_fn vec4 mul_vec4_rmat4(const vec4 a, const rmat4 b) {
    __m128 _Tmp3, _Tmp2, _Tmp1, _Tmp0;                                
    _Tmp0   = _mm_shuffle_ps(b.m0, b.m1, 0x44);
    _Tmp2   = _mm_shuffle_ps(b.m0, b.m1, 0xEE);
    _Tmp1   = _mm_shuffle_ps(b.m2, b.m3, 0x44);
    _Tmp3   = _mm_shuffle_ps(b.m2, b.m3, 0xEE);
    return create_vec4(
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88), 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD), 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88), 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a, _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD), 0b11110001))
    );
}

pure_fn vec4 mul_rmat4_vec4(const rmat4 a, const vec4 b) {
    return create_vec4(
        _mm_cvtss_f32(_mm_dp_ps(a.m0, b, 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m1, b, 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m2, b, 0b11110001)),
        _mm_cvtss_f32(_mm_dp_ps(a.m3, b, 0b11110001))
    );
}

pure_fn rmat4 mul_rmat4(const rmat4 a, const rmat4 b) {
    return (rmat4) {
        .m0 = _mm_fma_mac(
            _mm_permute_mac(a.m0, _MM_SHUFFLE(0, 0, 0, 0)), b.m0,
            _mm_fma_mac(
                _mm_permute_mac(a.m0, _MM_SHUFFLE(1, 1, 1, 1)), b.m1,
                _mm_fma_mac(
                    _mm_permute_mac(a.m0, _MM_SHUFFLE(2, 2, 2, 2)), b.m2,
                    _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(3, 3, 3, 3)), b.m3)
                )
            )
        ),
        .m1 = _mm_fma_mac(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 0, 0, 0)), b.m0,
            _mm_fma_mac(
                _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 1, 1, 1)), b.m1,
                _mm_fma_mac(
                    _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 2, 2, 2)), b.m2,
                    _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(3, 3, 3, 3)), b.m3)
                )
            )
        ),
        .m2 = _mm_fma_mac(
            _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 0, 0, 0)), b.m0,
            _mm_fma_mac(
                _mm_permute_mac(a.m2, _MM_SHUFFLE(1, 1, 1, 1)), b.m1,
                _mm_fma_mac(
                    _mm_permute_mac(a.m2, _MM_SHUFFLE(2, 2, 2, 2)), b.m2,
                    _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(3, 3, 3, 3)), b.m3)
                )
            )
        ),
        .m3 = _mm_fma_mac(
            _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 0, 0, 0)), b.m0,
            _mm_fma_mac(
                _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 1, 1, 1)), b.m1,
                _mm_fma_mac(
                    _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 2, 2, 2)), b.m2,
                    _mm_mul_ps(_mm_permute_mac(a.m3, _MM_SHUFFLE(3, 3, 3, 3)), b.m3)
                )
            )
        )
    };
}

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
    // + - +-   + - +-
    // 3 1 2    4 6 5
    // 0 2 3    6 3 2
    // 3 0 1    1 5 3
    // 0 1 2    4 2 1
    //          A B C
    // block (3030, A) will be _mm_addsub_ps
    __m128 dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    __m128 dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m3 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m2 = _mm_fms_mac(
        _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m3, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    // calulate determinants again, but with m2, m3
    dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m1 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m0 = _mm_fms_mac(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    return ret;
}

pure_fn rmat4 adj_rmat4(const rmat4 a) {
    // for the algorithm, refer to cofactor_mat4. adj_mat4 simply
    // uses that algorithm and then transposes that matrix
    rmat4 ret;
    
    __m128 dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    __m128 dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m3 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m2 = _mm_fms_mac(
        _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m3, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    // calulate determinants again, but with m2, m3
    dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m1 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m0 = _mm_fms_mac(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    // reuse dets_A, dets_B, dets_C as temp variables
    __m128 _Tmp3;
    _Tmp3   = _mm_shuffle_ps(ret.m2, ret.m3, 0xEE);
    dets_A   = _mm_shuffle_ps(ret.m0, ret.m1, 0xEE);
    dets_B   = _mm_shuffle_ps(ret.m2, ret.m3, 0x44);
    dets_C   = _mm_shuffle_ps(ret.m0, ret.m1, 0x44);

    ret.m0 = _mm_shuffle_ps(dets_C, dets_B, 0x88);
    ret.m1 = _mm_shuffle_ps(dets_C, dets_B, 0xDD);
    ret.m2 = _mm_shuffle_ps(dets_A, _Tmp3, 0x88);
    ret.m3 = _mm_shuffle_ps(dets_A, _Tmp3, 0xDD);

    return ret;
}

pure_fn rmat4 inv_rmat4(const rmat4 a) {
    rmat4 ret;
    
    __m128 dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    __m128 dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m0, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m1, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    __m128 dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m3 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m2 = _mm_fms_mac(
        _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m3, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    // calulate determinants again, but with m2, m3
    dets_A = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(1, 0, 2, 1)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 1, 3, 2)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 1, 3, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(1, 0, 2, 1)))
    );
    dets_B = _mm_fms_mac(
        _mm_permute_mac(a.m2, _MM_SHUFFLE(0, 1, 0, 2)), _mm_permute_mac(a.m3, _MM_SHUFFLE(2, 3, 3, 3)),
        _mm_mul_ps(_mm_permute_mac(a.m2, _MM_SHUFFLE(2, 3, 3, 3)), _mm_permute_mac(a.m3, _MM_SHUFFLE(0, 1, 0, 2)))
    );
    dets_C = _mm_blend_ps(
        _mm_permute_mac(dets_A, _MM_SHUFFLE(2, 0, 0, 0)),
        _mm_permute_mac(dets_B, _MM_SHUFFLE(0, 1, 3, 2)),
        0b0111
    );
    // C - B +- A
    ret.m1 = _mm_addsub_ps(
        _mm_fms_mac(
            _mm_permute_mac(a.m0, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(1, 0, 2, 1)), dets_B)
        ),
        _mm_mul_ps(_mm_permute_mac(a.m0, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
    );
    // B - (C +- A)
    ret.m0 = _mm_fms_mac(
        _mm_permute_mac(a.m1, _MM_SHUFFLE(1, 0, 2, 1)), dets_B,
        _mm_fmas_mac(
            _mm_permute_mac(a.m1, _MM_SHUFFLE(2, 1, 3, 2)), dets_C,
            _mm_mul_ps(_mm_permute_mac(a.m1, _MM_SHUFFLE(0, 3, 0, 3)), dets_A)
        )
    );

    // reuse dets_A, dets_B, dets_C as temp variables
    __m128 inv_det = _mm_rcp_mac(_mm_dp_ps(a.m0, ret.m0, 0b11111111));
    __m128 _Tmp3;
    _Tmp3   = _mm_shuffle_ps(ret.m2, ret.m3, 0xEE);
    dets_A   = _mm_shuffle_ps(ret.m0, ret.m1, 0xEE);
    dets_B   = _mm_shuffle_ps(ret.m2, ret.m3, 0x44);
    dets_C   = _mm_shuffle_ps(ret.m0, ret.m1, 0x44);

    ret.m0 = _mm_mul_ps(_mm_shuffle_ps(dets_C, dets_B, 0x88), inv_det);
    ret.m1 = _mm_mul_ps(_mm_shuffle_ps(dets_C, dets_B, 0xDD), inv_det);
    ret.m2 = _mm_mul_ps(_mm_shuffle_ps(dets_A, _Tmp3, 0x88), inv_det);
    ret.m3 = _mm_mul_ps(_mm_shuffle_ps(dets_A, _Tmp3, 0xDD), inv_det);

    return ret;
}


pure_fn rmat4 sqr_rmat4(const rmat4 a) {
    return (rmat4) {
        .m0 = _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 0))), a.m0,
            _mm_fma_mac(
                _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 1))), a.m1,
                _mm_fma_mac(
                    _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 2))), a.m2,
                    _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m0, 3))), a.m3)
                )
            )
        ),
        .m1 = _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 0))), a.m0,
            _mm_fma_mac(
                _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 1))), a.m1,
                _mm_fma_mac(
                    _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 2))), a.m2,
                    _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m1, 3))), a.m3)
                )
            )
        ),
        .m2 = _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 0))), a.m0,
            _mm_fma_mac(
                _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 1))), a.m1,
                _mm_fma_mac(
                    _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 2))), a.m2,
                    _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m2, 3))), a.m3)
                )
            )
        ),
        .m3 = _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m3, 0))), a.m0,
            _mm_fma_mac(
                _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m3, 1))), a.m1,
                _mm_fma_mac(
                    _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m3, 2))), a.m2,
                    _mm_mul_ps(_mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a.m3, 3))), a.m3)
                )
            )
        )
    };
}

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







void store_rmat4(float *arr, const rmat4 a) { // arr must be at least 16 wide
    _mm_storeu_ps(arr, a.m0);
    _mm_storeu_ps(arr + 4, a.m1);
    _mm_storeu_ps(arr + 8, a.m2);
    _mm_storeu_ps(arr + 12, a.m3);
}

void print_rmat4(const rmat4 a) {
    _Alignas(16) float a_arr[16];
    memcpy(a_arr, &a, sizeof(rmat4));
    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        a_arr[0], a_arr[1], a_arr[2], a_arr[3],
        a_arr[4], a_arr[5], a_arr[6], a_arr[7],
        a_arr[8], a_arr[9], a_arr[10], a_arr[11],
        a_arr[12], a_arr[13], a_arr[14], a_arr[15]
    );
}

void store_vec4(float *arr, const vec4 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

void print_vec4(const vec4 a) {
    _Alignas(16) float a_arr[4];
    _mm_store_ps(a_arr, a);
    printf("%f %f %f %f\n\n", a_arr[0], a_arr[1], a_arr[2], a_arr[3]);
}




#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
