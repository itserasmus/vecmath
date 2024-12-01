#ifndef VEC_MATH_CONSTRUCTORS_H_SSE
#define VEC_MATH_CONSTRUCTORS_H_SSE
#include "vec_math_sse.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
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
pure_fn mat3 create_mat3(
    const float a, const float b, const float c,
    const float p, const float q, const float r,
    const float x, const float y, const float z) {
    mat3 mat;
    mat.m0 = _mm_set_ps(0.0f, c, b, a);
    mat.m1 = _mm_set_ps(0.0f, r, q, p);
    mat.m2 = _mm_set_ps(0.0f, z, y, x);
    return mat;
}

pure_fn rmat4 create_rmat4(
    const float a, float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    rmat4 mat;
    mat.m0 = _mm_set_ps(d, c, b, a);
    mat.m1 = _mm_set_ps(k, j, i, h);
    mat.m2 = _mm_set_ps(s, r, q, p);
    mat.m3 = _mm_set_ps(w, z, y, x);
    return mat;
}


pure_fn mat4 create_mat4(
    const float a, float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    mat4 mat;
    mat.b0 = _mm_set_ps(i, h, b, a);
    mat.b1 = _mm_set_ps(k, j, d, c);
    mat.b2 = _mm_set_ps(y, x, q, p);
    mat.b3 = _mm_set_ps(w, z, s, r);
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
