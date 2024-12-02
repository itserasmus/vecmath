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
/************************************************************************
 *                        `mat4` or `rmat4`?                            *
 * -------------------------------------------------------------------- *
 * The choice between `mat4` and `rmat4` depends on the use case. Both  *
 * formats are highly optimized, but have differences that make them    *
 * better suited for different operations and scenarios:                *
 *                                                                      *
 * Use `mat4` when:                                                     *
 * - Faster performance for most mathematical operations is needed.     *
 * - The matrix is used mostly within VecMath or libraries that support *
 *   block matrices.                                                    *
 * - Conversion between block and row-major matrices is minimal.        *
 * - The matrix is short lived and mainly used within VecMath.          *
 * - Row-major conversions for compatibility with other libraries or    *
 *   systems is not frequent.                                           *
 *                                                                      *
 * Use `rmat4` when:                                                    *
 * - The matrix will be shared with other libraries or systems (e.g.,   *
 *   graphics APIs) that do not support block matrices.                 *
 * - Determinant and cofactor calculations are needed very frequently,  *
 *   as `rmat4` performs these slightly faster.                         *
 * - Conversion overhead between block and row-major matrices is large. *
 * - Low-level modifications are required, and block matrices are       *
 *   problematic.                                                       *
 *                                                                      *
 * Notes:                                                               *
 * - `mat4` performs better for most cases due to its symmetrical       *
 *   storage, but requires conversion for row-major compatibility.      *
 * - For temporary or short-lived matrices in graphics workflows,       *
 *   `rmat4` avoids conversion overhead.                                *
 * - Conversion between the two formats can be done using               *
 *   `cvt_mat4_rmat4` and `cvt_rmat4_mat4`.                             *
 ************************************************************************/




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
/// vectors (rows). Each vector (row) is represented by a `__m128`.
/// The last element of each vector is undefined, not necessarily 0.
/// They may be modified by VecMath and other functions without warning.
pure_fn mat3 create_mat3v(
    const __m128 a, const __m128 b, const __m128 c) {
    mat3 mat;
    mat.m0 = a;
    mat.m1 = b;
    mat.m2 = c;
    return mat;
}

/// @brief Creates a 3x3 row-major matrix from 9 float components.
/// @details This function creates a 3x3 matrix stored as 3 3-element
/// vectors (rows). Each vector (row) is represented by a `__m128`.
/// The last element of each vector is undefined, not necessarily 0.
/// They may be modified by VecMath and other functions without warning.
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

/// @brief Creates a 4x4 row-major matix from 4 4-element vectors
/// @details This function creates a 4x4 matrix stored as 4 4-element
/// vectors (rows). Each vector (row) is represented by a `__m128`.
/// This storage format performs worse than the `mat4` format for
/// most cases.
///
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics. For a detailed explanation,
/// see the comment at the top of this file.
pure_fn rmat4 create_rmat4v(
    const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    rmat4 mat;
    mat.m0 = a;
    mat.m1 = b;
    mat.m2 = c;
    mat.m3 = d;
    return mat;
}

/// @brief Creates a 4x4 row-major matix from 16 float components.
/// @details This function creates a 4x4 matrix stored as 4 4-element
/// vectors (rows). Each vector (row) is represented by a `__m128`.
/// This storage format performs worse than the `mat4` format for
/// most cases.
///
/// @attention This `rmat4` is a row-major matrix which can be safely
/// used with other libraries or graphics that use row-major matrices.
/// This matrix is preferable for short-lived/temporary matrices
/// shared with other libraries/graphics. For a detailed explanation,
/// see the comment at the top of this file.
pure_fn rmat4 create_rmat4(
    const float a, const float b, const float c, const float d,
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

/// @brief Creates a 4x4 block matrix from 16 float components.
/// @details This function creates a 4x4 matrix stored as 4 2x2 
/// blocks of 4 elements each. Each block is represented by a
/// `__m128`. This storage format performs better than the `rmat4`
/// for most cases.
/// 
/// @attention This `mat4` is a block matrix and needs to be
/// handled accordingly when performing low level manipulation or
/// using it with other libraries. To convert a `mat4` to a row
/// major format, use the `cvt_mat4_rmat4`. Although `mat4` usually
/// performs better than the `rmat4`, for short-lived/temporary
/// matrices shared with other libraries/graphics, consider using
/// the `rmat4`. For a detailed explanation, see the comment at the
/// top of this file.
pure_fn mat4 create_mat4(
    const float a, const float b, const float c, const float d,
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
