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


/// @brief Adds two 2-vectors.
pure_fn vec2 add_vec2(const vec2 a, const vec2 b) {
    return _mm_add_ps(a, b);
}

/// @brief Subtracts two 2-vectors.
pure_fn vec2 sub_vec2(const vec2 a, const vec2 b) {
    return _mm_sub_ps(a, b);
}

/// @brief Multiplies a 2-vector by a float.
pure_fn vec2 scalar_mul_vec2(const vec2 a, const float b) {
    return _mm_mul_ps(a, _mm_set1_ps(b));
}

/// @brief Multiplies two 2-vectors element-wise.
pure_fn vec2 mul_vec2(const vec2 a, const vec2 b) {
    return _mm_mul_ps(a, b);
}

/// @brief Divides a 2-vector by a float.
pure_fn vec2 scalar_div_vec2(const vec2 a, const float b) {
    return _mm_div_mac(a, _mm_set1_ps(b));
}

/// @brief Divides two 2-vectors element-wise.
pure_fn vec2 div_vec2(const vec2 a, const vec2 b) {
    return _mm_div_mac(a, b);
}

/// @brief Computes the dot product of two 2-vectors.
pure_fn float dot_vec2(const vec2 a, const vec2 b) {
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0b00110001));
}

/// @brief Normalizes a 2-vector.
pure_fn vec2 norm_vec2(const vec2 a) {
    __m128 inv_len = _mm_rsqrt_mac(_mm_dp_ps(a, a, 0b00110011));
    return _mm_mul_ps(a, inv_len);
}

/// @brief Computes the length of a 2-vector.
pure_fn float len_vec2(const vec2 a) {
    return sqrt(_mm_cvtss_f32(_mm_dp_ps(a, a, 0b00110001)));
}

/// @brief Computes the outer product of two 2-vectors.
pure_fn mat2 outer_vec2(const vec2 a, const vec2 b) {
    return _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(1, 1, 0, 0)),
        _mm_permute_mac(b, _MM_SHUFFLE(1, 0, 1, 0))
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
