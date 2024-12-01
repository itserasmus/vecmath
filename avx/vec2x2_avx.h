#ifndef VEC2X2_H_AVX
#define VEC2X2_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif


// add, sub, mul, div, dot, normalize, length, outer

pure_fn vec2 add_vec2(const vec2 a, const vec2 b) {
    return _mm_add_ps(a, b);
}

pure_fn vec2 sub_vec2(const vec2 a, const vec2 b) {
    return _mm_sub_ps(a, b);
}

pure_fn vec2 scalar_mul_vec2(const vec2 a, const float b) {
    return _mm_mul_ps(a, _mm_set1_ps(b));
}

pure_fn vec2 mul_vec2(const vec2 a, const vec2 b) {
    return _mm_mul_ps(a, b);
}

pure_fn vec2 scalar_div_vec2(const vec2 a, const float b) {
    return _mm_div_mac(a, _mm_set1_ps(b));
}

pure_fn vec2 div_vec2(const vec2 a, const vec2 b) {
    return _mm_div_mac(a, b);
}

pure_fn float dot_vec2(const vec2 a, const vec2 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b00110001));
}

pure_fn vec2 norm_vec2(const vec2 a) {
    float len = sqrt(dot_vec2(a, a));
    return scalar_div_vec2(a, len);
}

pure_fn float len_vec2(const vec2 a) {
    return sqrt(dot_vec2(a, a));
}

pure_fn mat2 outer_vec2(const vec2 a, const vec2 b) {
    mat2 mat1 = _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 0, 0));
    mat2 mat2 = _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 1, 0));
    return _mm_mul_ps(mat1, mat2);

}


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
