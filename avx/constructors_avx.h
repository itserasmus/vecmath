#ifndef VEC_MATH_CONSTRUCTORS_H_AVX
#define VEC_MATH_CONSTRUCTORS_H_AVX
#include "vec_math_avx.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx {
#endif
extern "C" {
#endif



pure_fn vec2 create_vec2(
    const float x, const float y) {
    return _mm_set_ps(0.0f, 0.0f, y, x);
}
pure_fn vec3 create_vec3(
    const float x, const float y, const float z) {
    return _mm_set_ps(0.0f, z, y, x);
}
pure_fn vec4 create_vec4(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}

pure_fn mat2 create_mat2(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}

pure_fn mat3 create_mat3v(
    const __m128 a, const __m128 b, const __m128 c) {
    mat3 mat;
    mat.m0 = _mm256_set_m128(b, a);
    mat.m1 = c;
    return mat;
}

pure_fn mat3 create_mat3(
    const float a, const float b, const float c,
    const float p, const float q, const float r,
    const float x, const float y, const float z) {
    mat3 mat;
    mat.m0 = _mm256_set_ps(0.0f, r, q, p, 0.0f, c, b, a);
    mat.m1 = _mm_set_ps(0.0f, z, y, x);
    return mat;
}

pure_fn rmat4 create_rmat4v256(
    const __m256 a, const __m256 b) {
    rmat4 mat;
    mat.m0 = a;
    mat.m1 = b;
    return mat;
}

pure_fn rmat4 create_rmat4v(
    const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    rmat4 mat;
    mat.m0 = _mm256_set_m128(b, a);
    mat.m1 = _mm256_set_m128(d, c);
    return mat;
}

pure_fn rmat4 create_rmat4(
    const float a, float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    rmat4 mat;
    mat.m0 = _mm256_set_ps(k, j, i, h, d, c, b, a);
    mat.m1 = _mm256_set_ps(w, z, y, x, s, r, q, p);
    return mat;
}


pure_fn mat4 create_mat4(
    const float a, const float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    mat4 mat;
    mat.b0 = _mm256_setr_ps(a, b, h, i,
                            r, s, z, w);
    mat.b1 = _mm256_setr_ps(c, d, j, k,
                            p, q, x, y);
    return mat;
}


#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
