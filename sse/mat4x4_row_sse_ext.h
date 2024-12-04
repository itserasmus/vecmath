/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Contains the `rmat4` related extensions for the SSE implementation of VecMath.
 */
#ifndef MAT4X4_ROW_EXT_H_SSE
#define MAT4X4_ROW_EXT_H_SSE
#include "vec_math_sse.h"
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif



// translate, rotate (vec, rad), scale, ortho, perspective, lookAt, rodriguez rotation
// projection: oblique, infinite perspective, reverse z
// reflective camera, stereo view

/// @brief Create a translation matrix from a vector.
/// @param b The vector to translate by.
pure_fn rmat4 translate_rmat4(const __m128 b) {
    rmat4 ret;
    __m128 b2 = _mm_blend_ps(b, _mm_setzero_ps(), 0b1000);
    ret.m0 = _mm_set_ps(_mm_extractf_ps(b, 0), 0.0f, 0.0f, 1.0f);
    ret.m1 = _mm_set_ps(_mm_extractf_ps(b, 1), 0.0f, 1.0f, 0.0f);
    ret.m2 = _mm_set_ps(_mm_extractf_ps(b, 2), 1.0f, 0.0f, 0.0f);
    ret.m3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return ret;
}

#ifdef VECM_RIGHT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Create a rotation matrix from a vector and an angle in a right-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn rmat4 rotate_rmat4_right_handed(const vec3 a, const float angle)
#else
/// @brief Create a rotation matrix from a vector and an angle in a right-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn rmat4 rotate_rmat4(const vec3 a, const float angle)
#endif
{
    // cos_vec (c) = [cosA+x^2(1-cosA)  cosA+y^2(1-cosA)  cosA+y^2(1-cosA)  - ]
    // sin_vec (s) = [xsinA  ysinA  zsinA  xsinA]
    // neg_cos_vec (n) = [yz(1-cosA)  xz(1-cosA)  xy(1-cosA)  yz(1-cosA)]
    // mat = |   c0   n2-s2 n1+s1    0  |
    //       | n2+s2    c1  n3-s3    0  |
    //       | n1-s1  n0+s0   c2     0  |
    //       |   0      0     0      1  |
    float sin = sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 tmp0 = _mm_permute_mac(a, _MM_SHUFFLE(0, 2, 1, 0));
    __m128 cos_vec = _mm_blend_ps(_mm_setzero_ps(), _mm_add_ps(
        _mm_set_ps1(cosf(angle)),
        _mm_mul_ps(_mm_mul_ps(tmp0, tmp0), _mm_set_ps1(neg_cos))
    ), 0b0111);
    __m128 sin_vec = _mm_mul_ps(tmp0, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(2, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );
    tmp0 = _mm_addsub_ps(neg_cos_vec, _mm_sub_ps(_mm_setzero_ps(), sin_vec)); // reuse variable
    sin_vec = _mm_addsub_ps(neg_cos_vec, sin_vec);

    rmat4 ret;
    ret.m3 = _mm_shuffle_ps(cos_vec, sin_vec, _MM_SHUFFLE(1, 2, 3, 0)); // c0 0 2- 1+
    ret.m0 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 2, 0));

    ret.m3 = _mm_shuffle_ps(cos_vec, tmp0, _MM_SHUFFLE(3, 2, 3, 1));    // c1 0 2+ 3-
    ret.m1 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 0, 2));

    ret.m2 = _mm_shuffle_ps(tmp0, cos_vec, _MM_SHUFFLE(3, 2, 0, 1));    // 1- 0+ c2 0-
    
    ret.m3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    
    return ret;
}
#endif
#ifdef VECM_LEFT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Create a rotation matrix from a vector and an angle in a left-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn rmat4 rotate_rmat4_left_handed(const vec3 a, const float angle)
#else
/// @brief Create a rotation matrix from a vector and an angle in a left-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn rmat4 rotate_rmat4(const vec3 a, const float angle)
#endif
    {
    // cos_vec (c) = [cosA+x^2(1-cosA)  cosA+y^2(1-cosA)  cosA+y^2(1-cosA)  - ]
    // sin_vec (s) = [xsinA  ysinA  zsinA  xsinA]
    // neg_cos_vec (n) = [yz(1-cosA)  xz(1-cosA)  xy(1-cosA)  yz(1-cosA)]
    // mat = |   c0   n2-s2 n1+s1    0  |
    //       | n2+s2    c1  n3-s3    0  |
    //       | n1-s1  n0+s0   c2     0  |
    //       |   0      0     0      1  |
    float sin = -sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 tmp0 = _mm_permute_mac(a, _MM_SHUFFLE(0, 2, 1, 0));
    __m128 cos_vec = _mm_blend_ps(_mm_setzero_ps(), _mm_add_ps(
        _mm_set_ps1(cosf(angle)),
        _mm_mul_ps(_mm_mul_ps(tmp0, tmp0), _mm_set_ps1(neg_cos))
    ), 0b0111);
    __m128 sin_vec = _mm_mul_ps(tmp0, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(2, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );
    tmp0 = _mm_addsub_ps(neg_cos_vec, _mm_sub_ps(_mm_setzero_ps(), sin_vec)); // reuse variable
    sin_vec = _mm_addsub_ps(neg_cos_vec, sin_vec);

    rmat4 ret;
    ret.m3 = _mm_shuffle_ps(cos_vec, sin_vec, _MM_SHUFFLE(1, 2, 3, 0)); // c0 0 2- 1+
    ret.m0 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 2, 0));

    ret.m3 = _mm_shuffle_ps(cos_vec, tmp0, _MM_SHUFFLE(3, 2, 3, 1));    // c1 0 2+ 3-
    ret.m1 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 0, 2));

    ret.m2 = _mm_shuffle_ps(tmp0, cos_vec, _MM_SHUFFLE(3, 2, 0, 1));    // 1- 0+ c2 0-
    
    ret.m3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    
    return ret;
}
#endif

/// @brief Create a scaling matrix from three scaling factors.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn rmat4 scale_rmat4(const float sc_x, const float sc_y, const float sc_z) {
    rmat4 ret;
    ret.m0 = _mm_set_ps(0.0f, 0.0f, 0.0f, sc_x);
    ret.m1 = _mm_set_ps(0.0f, 0.0f, sc_y, 0.0f);
    ret.m2 = _mm_set_ps(0.0f, sc_z, 0.0f, 0.0f);
    ret.m3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return ret;
}

/// @brief Creates a reflection matrix across a plane given by a normal vector.
/// @param n The normal vector of the plane to reflect across.
pure_fn rmat4 reflect_rmat4(const vec3 n) {
    // | xx xy xz 0 |
    // | yx yy yz 0 |
    // | zx zy zz 0 |
    // | 0  0  0  1 |
    rmat4 ret;
    ret.m0 = _mm_sub_ps(
        _mm_set_ps(0.0f, 0.0f, 0.0f, 0.5f),
        _mm_mul_ps(n, _mm_permute_mac(n, _MM_SHUFFLE(0, 0, 0, 0)))
    ); // xx xy xz 0
    ret.m0 = _mm_blend_ps(_mm_add_ps(ret.m0, ret.m0), _mm_setzero_ps(), 0b1000);
    __m128 prod = _mm_sub_ps(
        _mm_set_ps(0.0f, 0.0f, 0.5f, 0.5f),
        _mm_mul_ps(_mm_permute_mac(n, _MM_SHUFFLE(0, 1, 2, 1)),
        _mm_permute_mac(n, _MM_SHUFFLE(0, 2, 2, 1)))
    ); // yy zz yz 0
    prod = _mm_blend_ps(_mm_add_ps(prod, prod), _mm_setzero_ps(), 0b1000);
    print_vec4(prod);
    
    ret.m3 = _mm_shuffle_ps(prod, ret.m0, _MM_SHUFFLE(3, 1, 2, 0)); // yy yz yx 0
    ret.m1 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(3, 1, 0, 2));
    ret.m3 = _mm_shuffle_ps(prod, ret.m0, _MM_SHUFFLE(3, 2, 2, 1)); // zz zy zx 0
    ret.m2 = _mm_permute_mac(ret.m3, _MM_SHUFFLE(3, 1, 2, 0));
    ret.m3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return ret;
}

/// @brief Create a pererspective projection matrix.
/// @param aspect The aspect ratio of the viewport.
/// @param fov The field of view in radians.
/// @param _near The distance to the near plane.
/// @param _far The distance to the far plane.
pure_fn rmat4 perspective_rmat4(const float aspect, const float fov, const float _near, const float _far) {
    // _near and _far since Win32 defines near and far as macros
    float inv_asp_tanfov = 1.0f/(aspect * tanf(fov/2));
    float inv_near_minus_far = 1.0f/(_near - _far);
    rmat4 ret;
    ret.m0 = _mm_set_ps(0.0f, 0.0f, 0.0f, inv_asp_tanfov);
    ret.m1 = _mm_set_ps(0.0f, 0.0f, inv_asp_tanfov*aspect, 0.0f);
    ret.m2 = _mm_set_ps(2*(_far*_near)*inv_near_minus_far, 0.0f, (_far + _near)*inv_near_minus_far, 0.0f);
    ret.m3 = _mm_set_ps(0.0f, -1.0f, 0.0f, 0.0f);
    return ret;
}

/// @brief Create a look-at matrix.
/// @param camera The position of the camera.
/// @param target The position of the target.
/// @param up The up vector.
pure_fn rmat4 look_at_rmat4(const vec3 camera, const vec3 target, const vec3 up) {
    // m0 = right, m1 = up, m2 = -forward
    __m128 neg_cam = _mm_xor_ps(camera, _mm_set_ps1(-0.0f));
    rmat4 ret;
    // -forward = normalize(cam-target)
    ret.m2 = _mm_sub_ps(camera, target);
    ret.m2 = _mm_mul_ps(ret.m2, _mm_rsqrt_mac(_mm_dp_ps(ret.m2, ret.m2, 0b01110111)));
    ret.m2 = _mm_blend_ps(ret.m2, _mm_dp_ps(ret.m2, neg_cam, 0b01111111), 0b1000);

    // right = normalize(up x -forward)
    ret.m0 = _mm_fms_mac(
        _mm_permute_mac(ret.m2, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(up, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_mul_ps(
            _mm_permute_mac(ret.m2, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_permute_mac(up, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    ret.m0 = _mm_mul_ps(ret.m0, _mm_rsqrt_mac(_mm_dp_ps(ret.m0, ret.m0, 0b01110111)));
    ret.m0 = _mm_blend_ps(ret.m0, _mm_dp_ps(ret.m0, neg_cam, 0b01111111), 0b1000);

    // up' = -forward x right
    ret.m1 = _mm_fms_mac(
        _mm_permute_mac(ret.m0, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(ret.m2, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_mul_ps(
            _mm_permute_mac(ret.m0, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_permute_mac(ret.m2, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    ret.m1 = _mm_blend_ps(ret.m1, _mm_dp_ps(ret.m1, neg_cam, 0b01111111), 0b1000);
    ret.m3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return ret;
}

/// @brief Rotate a vector around another vector by an angle.
/// @param a The vector to rotate.
/// @param axis The axis to rotate around.
/// @param angle The angle to rotate by in radians.
pure_fn vec3 rotate_vec3(const vec3 a, const vec3 axis, const float angle) {
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
    neg_cos = 1.0f - cosf(angle);
    #else
    float half_sin = sinf(angle/2); // for numerical stability when cosA is close to 1
    neg_cos = 2.0f*half_sin*half_sin;
    #endif
    return _mm_fma_mac(
        a, _mm_set_ps1(cos_angle),
        _mm_fma_mac(
            _mm_fms_mac(
                _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 2, 1)),
                _mm_permute_mac(axis, _MM_SHUFFLE(0, 1, 0, 2)),
                _mm_mul_ps(
                    _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 0, 2)),
                    _mm_permute_mac(axis, _MM_SHUFFLE(0, 0, 2, 1))
                )
            ),
            _mm_set_ps1(sin_angle),
            _mm_mul_ps(axis, _mm_mul_ps(_mm_set_ps1(neg_cos), _mm_dp_ps(a, axis, 0b01111111)))
        )
    );
}


#ifdef VECM_RIGHT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a right-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn rmat4 affine_rmat4_right_handed(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#else
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a right-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn rmat4 affine_rmat4(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#endif
{
    // cos_vec (c) = [cosA+x^2(1-cosA)  cosA+y^2(1-cosA)  cosA+y^2(1-cosA)  - ]
    // sin_vec (s) = [xsinA  ysinA  zsinA  xsinA]
    // neg_cos_vec (n) = [yz(1-cosA)  xz(1-cosA)  xy(1-cosA)  yz(1-cosA)]
    // mat = |   c0   n2-s2 n1+s1    0  |
    //       | n2+s2    c1  n3-s3    0  |
    //       | n1-s1  n0+s0   c2     0  |
    //       |   0      0     0      1  |
    float sin = sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 tmp0 = _mm_permute_mac(axis, _MM_SHUFFLE(0, 2, 1, 0));
    __m128 cos_vec = _mm_blend_ps(_mm_setzero_ps(), _mm_add_ps(
        _mm_set_ps1(cosf(angle)),
        _mm_mul_ps(_mm_mul_ps(tmp0, tmp0), _mm_set_ps1(neg_cos))
    ), 0b0111);
    __m128 sin_vec = _mm_mul_ps(tmp0, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(axis, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(axis, _MM_SHUFFLE(2, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );
    tmp0 = _mm_addsub_ps(neg_cos_vec, _mm_sub_ps(_mm_setzero_ps(), sin_vec)); // reuse variable
    sin_vec = _mm_addsub_ps(neg_cos_vec, sin_vec);

    rmat4 ret;
    ret.m3 = _mm_shuffle_ps(cos_vec, sin_vec, _MM_SHUFFLE(1, 2, 3, 0)); // c0 0 2- 1+
    ret.m0 = _mm_blend_ps(
        _mm_mul_ps(_mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 2, 0)), _mm_set_ps1(sc_x)),
        _mm_permute_mac(translation, _MM_SHUFFLE(0, 0, 0, 0)),
        0b1000
    );

    ret.m3 = _mm_shuffle_ps(cos_vec, tmp0, _MM_SHUFFLE(3, 2, 3, 1));    // c1 0 2+ 3-
    ret.m1 = _mm_blend_ps(
        _mm_mul_ps(_mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 0, 2)), _mm_set_ps1(sc_y)),
        _mm_permute_mac(translation, _MM_SHUFFLE(1, 1, 1, 1)),
        0b1000
    );

    ret.m2 = _mm_blend_ps(
        _mm_mul_ps(_mm_shuffle_ps(tmp0, cos_vec, _MM_SHUFFLE(3, 2, 0, 1)), _mm_set_ps1(sc_z)),    // 1- 0+ c2 0
        _mm_permute_mac(translation, _MM_SHUFFLE(3, 3, 3, 3)),
        0b1000
    );
    
    ret.m3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    
    return ret;
}
#endif
#ifdef VECM_LEFT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a left-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn rmat4 affine_rmat4_left_handed(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#else
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a left-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn rmat4 affine_rmat4(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#endif
{
    // cos_vec (c) = [cosA+x^2(1-cosA)  cosA+y^2(1-cosA)  cosA+y^2(1-cosA)  - ]
    // sin_vec (s) = [xsinA  ysinA  zsinA  xsinA]
    // neg_cos_vec (n) = [yz(1-cosA)  xz(1-cosA)  xy(1-cosA)  yz(1-cosA)]
    // mat = |   c0   n2-s2 n1+s1    0  |
    //       | n2+s2    c1  n3-s3    0  |
    //       | n1-s1  n0+s0   c2     0  |
    //       |   0      0     0      1  |
    float sin = -sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 tmp0 = _mm_permute_mac(axis, _MM_SHUFFLE(0, 2, 1, 0));
    __m128 cos_vec = _mm_blend_ps(_mm_setzero_ps(), _mm_add_ps(
        _mm_set_ps1(cosf(angle)),
        _mm_mul_ps(_mm_mul_ps(tmp0, tmp0), _mm_set_ps1(neg_cos))
    ), 0b0111);
    __m128 sin_vec = _mm_mul_ps(tmp0, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(axis, _MM_SHUFFLE(1, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(axis, _MM_SHUFFLE(2, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );
    tmp0 = _mm_addsub_ps(neg_cos_vec, _mm_sub_ps(_mm_setzero_ps(), sin_vec)); // reuse variable
    sin_vec = _mm_addsub_ps(neg_cos_vec, sin_vec);

    rmat4 ret;
    ret.m3 = _mm_shuffle_ps(cos_vec, sin_vec, _MM_SHUFFLE(1, 2, 3, 0)); // c0 0 2- 1+
    ret.m0 = _mm_blend_ps(
        _mm_mul_ps(_mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 2, 0)), _mm_set_ps1(sc_x)),
        _mm_permute_mac(translation, _MM_SHUFFLE(0, 0, 0, 0)),
        0b1000
    );

    ret.m3 = _mm_shuffle_ps(cos_vec, tmp0, _MM_SHUFFLE(3, 2, 3, 1));    // c1 0 2+ 3-
    ret.m1 = _mm_blend_ps(
        _mm_mul_ps(_mm_permute_mac(ret.m3, _MM_SHUFFLE(1, 3, 0, 2)), _mm_set_ps1(sc_y)),
        _mm_permute_mac(translation, _MM_SHUFFLE(1, 1, 1, 1)),
        0b1000
    );

    ret.m2 = _mm_blend_ps(
        _mm_mul_ps(_mm_shuffle_ps(tmp0, cos_vec, _MM_SHUFFLE(3, 2, 0, 1)), _mm_set_ps1(sc_z)),    // 1- 0+ c2 0
        _mm_permute_mac(translation, _MM_SHUFFLE(3, 3, 3, 3)),
        0b1000
    );
    
    ret.m3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    
    return ret;
}
#endif














#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif
