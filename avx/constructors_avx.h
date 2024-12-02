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


/// @brief Creates a 2-component vector from 2 float components.
/// @details This function creates a 2-component vector stored as a
/// single `__m128`. The last two elements are undefined, not
/// necessarily 0. They may be modified by VecMath and other functions
/// without warning.
pure_fn vec2 create_vec2(
    const float x, const float y) {
    return _mm_set_ps(0.0f, 0.0f, y, x);
}
/// @brief Creates a 3-component vector from 3 float components.
/// @details This function creates a 3-component vector stored as a
/// single `__m128`. The last element is undefined, not necessarily 0.
/// It may be modified by VecMath and other functions without warning.
pure_fn vec3 create_vec3(
    const float x, const float y, const float z) {
    return _mm_set_ps(0.0f, z, y, x);
}
/// @brief Creates a 4-component vector from 4 float components.
/// @details This function creates a 4-component vector stored as a
/// single `__m128`.
pure_fn vec4 create_vec4(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}

/// @brief Creates a 2x2 row-major matrix from 4 float components.
/// @details This function creates a 2x2 matrix stored as a single
/// `__m128`. The first two elements are the first row of the matrix,
/// and the last two elements are the second row of the matrix.
pure_fn mat2 create_mat2(
    const float x, const float y, const float z, const float w) {
    return _mm_set_ps(w, z, y, x);
}

/// @brief Creates a 3x3 row-major matrix from three 3-element vectors.
/// @details This function creates a 3x3 matrix stored as 3 3-element
/// vectors (rows). The first two vectors (rows) are represented by the
/// two lanes of a `__m256`. and the third vector (row) is represented
/// by a `__m128`. The last element of each vector is undefined, not
/// necessarily 0. They may be modified by VecMath and other functions
/// without warning.
pure_fn mat3 create_mat3v(
    const __m128 a, const __m128 b, const __m128 c) {
    mat3 mat;
    mat.m0 = _mm256_set_m128(b, a);
    mat.m1 = c;
    return mat;
}

/// @brief Creates a 3x3 row-major matrix from 9 float components.
/// @details This function creates a 3x3 matrix stored as a 6-element
/// vector (row) and a 3-element vector (row). The first two vectors
/// (rows) are represented by the two lanes of a `__m256`. and the third
/// vector (row) is represented by a `__m128`. The last element of each
/// vector is undefined, not necessarily 0. They may be modified by
/// VecMath and other functions without warning.
pure_fn mat3 create_mat3(
    const float a, const float b, const float c,
    const float p, const float q, const float r,
    const float x, const float y, const float z) {
    mat3 mat;
    mat.m0 = _mm256_set_ps(0.0f, r, q, p, 0.0f, c, b, a);
    mat.m1 = _mm_set_ps(0.0f, z, y, x);
    return mat;
}

/// @brief Creates a 4x4 row-major matix from 2 8-element vectors.
/// @details This function creates a 4x4 matrix stored as 2 8-element
/// vectors (rows). Each pair of vectors (rows) is represented by a
/// `__m256`. This storage format performs worse than the `mat4` format for
/// most cases.
///
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
pure_fn rmat4 create_rmat4v256(
    const __m256 a, const __m256 b) {
    rmat4 mat;
    mat.m0 = a;
    mat.m1 = b;
    return mat;
}

/// @brief Creates a 4x4 row-major matix from 4 4-element vectors.
/// @details This function creates a 4x4 matrix stored as 2 8-element
/// vectors (rows). Each pair of vectors (rows) is represented by a
/// `__m256`. This storage format performs worse than the `mat4` format for
/// most cases.
///
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
pure_fn rmat4 create_rmat4v(
    const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    rmat4 mat;
    mat.m0 = _mm256_set_m128(b, a);
    mat.m1 = _mm256_set_m128(d, c);
    return mat;
}

/// @brief Creates a 4x4 row-major matix from 16 float components.
/// @details This function creates a 4x4 matrix stored as 2 8-element
/// vectors (rows). Each pair of vectors (rows) is represented by a
/// `__m256`. This storage format performs worse than the `mat4` format for
/// most cases.
///
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics.
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

/// @brief Creates a 4x4 block matrix from 16 float components.
/// @details This function creates a 4x4 matrix stored as 2 pairs
// of 2x2 blocks of 4 elements each. Both pairs of diagonal blocks
/// is represented by a `__m256`. This storage format performs
/// better than the `rmat4` for most cases.
/// 
/// @attention This `mat4` is a block matrix and needs to be
/// handled accordingly when performing low level manipulation or
/// using it with other libraries. To convert a `mat4` to a row
/// major format, use the `cvt_mat4_rmat4`. Although `mat4` usually
/// performs better than the `rmat4`, for short-lived/temporary
/// matrices shared with other libraries/graphics, consider using
/// the `rmat4`.
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
