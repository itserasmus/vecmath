#ifndef VEC4X4_H_AVX
#define VEC4X4_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif


// add, sub, mul, div, dot, normalize, length, outer

pure_fn vec4 add_vec4(const vec4 a, const vec4 b) {
    return _mm_add_ps(a, b);
}

pure_fn vec4 sub_vec4(const vec4 a, const vec4 b) {
    return _mm_sub_ps(a, b);
}

pure_fn vec4 scalar_mul_vec4(const vec4 a, const float b) {
    return _mm_mul_ps(a, _mm_set_ps1(b));
}

pure_fn vec4 mul_vec4(const vec4 a, const vec4 b) {
    return _mm_mul_ps(a, b);
}

pure_fn vec4 scalar_div_vec4(const vec4 a, const float b) {
    return _mm_div_mac(a, _mm_set_ps1(b));
}

pure_fn vec4 div_vec4(const vec4 a, const vec4 b) {
    return _mm_div_mac(a, b);
}

pure_fn float dot_vec4(const vec4 a, const vec4 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b11110001));
}

pure_fn vec4 norm_vec4(const vec4 a) {
    float len = sqrt(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b11110001)));
    return _mm_div_mac(a, _mm_set_ps1(len));
}

pure_fn float len_vec4(const vec4 a) {
    return sqrt(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b11110001)));
}

pure_fn rmat4 outer_vec4r(const vec4 a, const vec4 b) {
    rmat4 ret;
    ret.m0 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1)),
            _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0))
        )
    );
    ret.m1 = _mm256_mul_ps(
        _mm256_set_m128(b, b),
        _mm256_set_m128(
            _mm_permute_mac(a, _MM_SHUFFLE(3, 3, 3, 3)),
            _mm_permute_mac(a, _MM_SHUFFLE(2, 2, 2, 2))
        )
    );
    return ret;
}






#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
