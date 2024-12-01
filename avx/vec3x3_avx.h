#ifndef VEC3X3_H_AVX
#define VEC3X3_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif



// add, sub, mul, div, dot, normalize, length, outer

pure_fn vec3 add_vec3(const vec3 a, const vec3 b) {
    return _mm_add_ps(a, b);
}

pure_fn vec3 sub_vec3(const vec3 a, const vec3 b) {
    return _mm_sub_ps(a, b);
}

pure_fn vec3 scalar_mul_vec3(const vec3 a, const float b) {
    return _mm_mul_ps(a, _mm_set_ps1(b));
}

pure_fn vec3 mul_vec3(const vec3 a, const vec3 b) {
    return _mm_mul_ps(a, b);
}

pure_fn vec3 scalar_div_vec3(const vec3 a, const float b) {
    return _mm_div_mac(a, _mm_set_ps1(b));
}

pure_fn vec3 div_vec3(const vec3 a, const vec3 b) {
    return _mm_div_mac(a, b);
}

pure_fn float dot_vec3(const vec3 a, const vec3 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b01110001));
}

pure_fn vec3 norm_vec3(const vec3 a) {
    float inv_len = 1.0f/sqrtf(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b01110001)));
    return _mm_mul_ps(a, _mm_set_ps1(inv_len));
}

pure_fn float len_vec3(const vec3 a) {
    return sqrtf(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b01110001)));
}

pure_fn mat3 outer_vec3(const vec3 a, const vec3 b) {
    mat3 ret;
    ret.m0 = _mm256_mul_ps(
        _mm256_set_m128(_mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 0)), _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 1, 1))),
        _mm256_set_m128(b, b)
    );
    ret.m1 = _mm_mul_ps(_mm_permute_ps(a, _MM_SHUFFLE(3, 3, 3, 3)), b);
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
