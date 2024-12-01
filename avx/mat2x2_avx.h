#ifndef MAT2X2_H_AVX
#define MAT2X2_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif



// add, sub, scal_mul, mul, det, adj, inv, trans, pre_vec_mul, post_vec_mul, powers

pure_fn mat2 add_mat2(const mat2 a, const mat2 b) {
    return _mm_add_ps(a, b);
}

pure_fn mat2 sub_mat2(const mat2 a, const mat2 b) {
    return _mm_sub_ps(a, b);
}

pure_fn mat2 scal_mul_mat2(const mat2 a, const float b) {
    return _mm_mul_ps(a, _mm_set1_ps(b));
}

pure_fn mat2 trans_mat2(const mat2 a) {
    return _mm_permute_mac(a, _MM_SHUFFLE(3, 1, 2, 0));
}

pure_fn float det_mat2(const mat2 a) {
    vec4 pairs = _mm_mul_ps(a, _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3)));
    vec4 sub = _mm_sub_ps(pairs, _mm_permute_mac(pairs, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(sub);
}

pure_fn vec2 mul_vec2_mat2(const vec2 a, const mat2 b) {
    __m128 prod = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 1, 0)),
        _mm_permute_mac(b, _MM_SHUFFLE(3, 1, 2, 0))
    );
    return _mm_hadd_ps(prod, prod);
}

pure_fn vec2 mul_mat2_vec2(const mat2 a, const vec2 b) {
    __m128 v_mul = _mm_mul_ps(a, _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 1, 0)));
    return _mm_hadd_ps(v_mul, v_mul);
}

pure_fn mat2 cofactor_mat2(const mat2 a) {
    mat2 aperm = _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3));
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), aperm, 0b0110);
    return _mm_sub_ps(aperm, _mm_add_ps(adj_neg, adj_neg));
}

pure_fn mat2 adj_mat2(const mat2 a) {
    mat2 aperm = _mm_permute_mac(a, _MM_SHUFFLE(0, 2, 1, 3));
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), aperm, 0b0110);
    return _mm_sub_ps(aperm, _mm_add_ps(adj_neg, adj_neg));
}

pure_fn mat2 inv_mat2(const mat2 a) {
    vec4 pairs = _mm_mul_ps(a, _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 3)));
    vec4 sub = _mm_sub_ps(pairs, _mm_permute_mac(pairs, _MM_SHUFFLE(0, 0, 0, 1)));
    float det = _mm_cvtss_f32(sub);
    mat2 adj_neg = _mm_blend_ps(_mm_setzero_ps(), a, 0b0110);
    mat2 adj_not_perm = _mm_sub_ps(a, _mm_add_ps(adj_neg, adj_neg));
    return _mm_div_mac(_mm_permute_mac(adj_not_perm, _MM_SHUFFLE(0, 2, 1, 3)), _mm_set1_ps(det));
}

pure_fn mat2 mul_mat2(const mat2 a, const mat2 b) {
    return _mm_fma_mac(
        _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 0, 0)), b,
        _mm_mul_ps(_mm_permute_mac(a, _MM_SHUFFLE(2, 2, 1, 1)), _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 3, 2)))
    );
}

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

pure_fn mat2 cube_mat2(const mat2 a) {
    //        (C0 + C1)C2+ ( C3 )C4C5
    // res0 = (aa + bc)a + (a + d)bc
    // res1 = (aa + bc)b + (a + d)bd
    // res2 = (dd + bc)c + (a + d)ac
    // res3 = (dd + bc)d + (a + d)bc
    __m128 C0 = _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 0, 0));
    C0 = _mm_mul_ps(C0, C0);
    __m128 accum = _mm_mul_ps(
        a,
        _mm_fma_mac(
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a, 1))),
            _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a, 2))),
            C0
        )
    );
    __m128 C3 = _mm_set_ps1(reinterpret_int_float(_mm_extract_ps(a, 0)) + reinterpret_int_float(_mm_extract_ps(a, 3)));
    return _mm_fma_mac(
        _mm_mul_ps( // C4C5
            _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 1, 1)),
            _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 3, 2))
        ),
        C3,
        accum
    );
}

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

void store_mat2(float *arr, const mat2 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

void print_mat2(const mat2 a) {
    printf("%f %f\n%f %f\n\n", _mm_extractf_ps(a, 0), _mm_extractf_ps(a, 1), _mm_extractf_ps(a, 2), _mm_extractf_ps(a, 3));
}

void store_vec2(float *arr, const vec2 a) { // arr must be at least 4 wide
    _mm_storeu_ps(arr, a);
}

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
