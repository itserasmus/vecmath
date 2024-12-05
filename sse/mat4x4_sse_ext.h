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
#ifndef MAT4X4_EXT_H_SSE
#define MAT4X4_EXT_H_SSE
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
pure_fn mat4 translate_mat4(const __m128 b) {
    mat4 ret;
    __m128 b2 = _mm_blend_ps(b, _mm_setzero_ps(), 0b1000);
    ret.b0 = _mm_set_ps(1.0f, 0.0f, 0.0f, 1.0f);
    ret.b1 = _mm_set_ps(_mm_extractf_ps(b, 1), 0.0f, _mm_extractf_ps(b, 0), 0.0f);
    ret.b2 = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    ret.b3 = _mm_set_ps(1.0f, 0.0f, _mm_extractf_ps(b, 2), 1.0f);
    return ret;
}

#ifdef VECM_RIGHT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Create a rotation matrix from a vector and an angle in a right-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn rmat4 rotate_mat4_right_handed(const vec3 a, const float angle)
#else
/// @brief Create a rotation matrix from a vector and an angle in a right-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn mat4 rotate_mat4(const vec3 a, const float angle)
#endif
{
    // refer to `rotate_rmat4` for algorithm
    float sin = sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 cos_vec = _mm_fma_mac(
        _mm_mul_ps(a, a),
        _mm_set_ps1(neg_cos),
        _mm_set_ps1(cosf(angle))
    );
    __m128 sin_vec = _mm_mul_ps(a, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );

    mat4 ret;
    ret.b3 = _mm_addsub_ps(
        _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 1, 2, 2)),
        _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 1, 2, 2))
    );
    sin_vec = _mm_addsub_ps(
        _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 0, 1, 0)),
        _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 0, 1, 0))
    ); // - 1- 0+ -
    ret.b0 = _mm_shuffle_ps(cos_vec, ret.b3, _MM_SHUFFLE(1, 0, 1, 0)); // c0 c1 2+ 2-
    ret.b0 = _mm_permute_mac(ret.b0, _MM_SHUFFLE(1, 3, 2, 0));

    ret.b1 = _mm_shuffle_ps(_mm_setzero_ps(), sin_vec, _MM_SHUFFLE(2, 1, 0, 0)); // 0 0 1+ 0-
    ret.b1 = _mm_permute_mac(ret.b1, _MM_SHUFFLE(0, 3, 0, 2));

    ret.b2 = _mm_shuffle_ps(ret.b3, _mm_setzero_ps(), _MM_SHUFFLE(0, 0, 3, 2));

    ret.b3 = _mm_set_ps(1.0f, 0.0f, 0.0f, _mm_extractf_ps(cos_vec, 2));
    
    
    return ret;
}
#endif
#ifdef VECM_LEFT_HANDED
#ifdef VECM_HANDEDNESS_NAMING
/// @brief Create a rotation matrix from a vector and an angle in a left-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn mat4 rotate_mat4_left_handed(const vec3 a, const float angle)
#else
/// @brief Create a rotation matrix from a vector and an angle in a left-handed system.
/// @param a The vector to rotate around.
/// @param angle The angle to rotate by.
pure_fn mat4 rotate_mat4(const vec3 a, const float angle)
#endif
{
    // refer to `rotate_rmat4` for algorithm
    float sin = -sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 cos_vec = _mm_fma_mac(
        _mm_mul_ps(a, a),
        _mm_set_ps1(neg_cos),
        _mm_set_ps1(cosf(angle))
    );
    __m128 sin_vec = _mm_mul_ps(a, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(a, _MM_SHUFFLE(0, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(a, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );

    mat4 ret;
    ret.b3 = _mm_addsub_ps(
        _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 1, 2, 2)),
        _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 1, 2, 2))
    );
    sin_vec = _mm_addsub_ps(
        _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 0, 1, 0)),
        _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 0, 1, 0))
    ); // - 1- 0+ -
    ret.b0 = _mm_shuffle_ps(cos_vec, ret.b3, _MM_SHUFFLE(1, 0, 1, 0)); // c0 c1 2+ 2-
    ret.b0 = _mm_permute_mac(ret.b0, _MM_SHUFFLE(1, 3, 2, 0));

    ret.b1 = _mm_shuffle_ps(_mm_setzero_ps(), sin_vec, _MM_SHUFFLE(2, 1, 0, 0)); // 0 0 1+ 0-
    ret.b1 = _mm_permute_mac(ret.b1, _MM_SHUFFLE(0, 3, 0, 2));

    ret.b2 = _mm_shuffle_ps(ret.b3, _mm_setzero_ps(), _MM_SHUFFLE(0, 0, 3, 2));

    ret.b3 = _mm_set_ps(1.0f, 0.0f, 0.0f, _mm_extractf_ps(cos_vec, 2));
    
    
    return ret;
}
#endif

/// @brief Create a scaling matrix from three scaling factors.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn mat4 scale_mat4(const float sc_x, const float sc_y, const float sc_z) {
    mat4 ret;
    ret.b0 = _mm_set_ps(sc_y, 0.0f, 0.0f, sc_x);
    ret.b1 = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    ret.b2 = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    ret.b3 = _mm_set_ps(1.0f, 0.0f, 0.0f, sc_z);
    return ret;
}

/// @brief Creates a reflection matrix across a plane given by a normal unit vector.
/// @param n The normal unit vector of the plane to reflect across.
pure_fn mat4 reflect_mat4(const vec3 n) {
    // I - 2*| xx xy xz 0 |
    //       | yx yy yz 0 |
    //       | zx zy zz 0 |
    //       | 0  0  0  0 |
    mat4 ret;
    __m128 tmp0 = _mm_mul_ps(n, n); // xx yy zz -
    tmp0 = _mm_sub_ps(_mm_set_ps1(1.0f), _mm_add_ps(tmp0, tmp0));
    __m128 tmp1 = _mm_mul_ps(
        _mm_permute_mac(n, _MM_SHUFFLE(0, 1, 0, 0)),
        _mm_permute_mac(n, _MM_SHUFFLE(0, 2, 2, 1))
    ); // xy xz yz -
    tmp1 = _mm_blend_ps(_mm_xor_ps(_mm_add_ps(tmp1, tmp1), _mm_set_ps1(-0.0f)), _mm_setzero_ps(), 0b1000);

    ret.b0 = _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(0, 0, 1, 0)); // xx yy xy yx
    ret.b0 = _mm_permute_mac(ret.b0, _MM_SHUFFLE(1, 2, 2, 0));

    ret.b1 = _mm_permute_mac(tmp1, _MM_SHUFFLE(3, 2, 3, 1));
    ret.b2 = _mm_permute_mac(tmp1, _MM_SHUFFLE(3, 3, 2, 1));
    
    ret.b3 = _mm_set_ps(1.0f, 0.0f, 0.0f, _mm_extractf_ps(tmp0, 2));
    
    return ret;
}

/// @brief Create a pererspective projection matrix.
/// @param aspect The aspect ratio of the viewport.
/// @param fov The field of view in radians.
/// @param _near The distance to the near plane.
/// @param _far The distance to the far plane.
pure_fn mat4 perspective_mat4(const float aspect, const float fov, const float _near, const float _far) {
    // _near and _far since Win32 defines near and far as macros
    float inv_asp_tanfov = 1.0f/(aspect * tanf(fov/2));
    float inv_near_minus_far = 1.0f/(_near - _far);
    mat4 ret;
    ret.b0 = _mm_set_ps(inv_asp_tanfov*aspect, 0.0f, 0.0f, inv_asp_tanfov);
    ret.b1 = _mm_set_ps1(0.0f);
    ret.b2 = _mm_set_ps(0.0f, 0.0f, (_far + _near)*inv_near_minus_far, 0.0f);
    ret.b3 = _mm_set_ps(0.0f, -1.0f, 2*(_far*_near)*inv_near_minus_far, 0.0f);
    return ret;
}

/// @brief Create a look-at matrix.
/// @param camera The position of the camera.
/// @param target The position of the target.
/// @param up The up vector.
pure_fn mat4 look_at_mat4(const vec3 camera, const vec3 target, const vec3 up) {
    // m0 = right, m1 = up, m2 = -forward
    __m128 neg_cam = _mm_xor_ps(camera, _mm_set_ps1(-0.0f));
    rmat4 tmp;
    // -forward = normalize(cam-target)
    tmp.m2 = _mm_sub_ps(camera, target);
    tmp.m2 = _mm_mul_ps(tmp.m2, _mm_rsqrt_mac(_mm_dp_ps(tmp.m2, tmp.m2, 0b01110111)));
    tmp.m2 = _mm_blend_ps(tmp.m2, _mm_dp_ps(tmp.m2, neg_cam, 0b01111111), 0b1000);

    // right = normalize(up x -forward)
    tmp.m0 = _mm_fms_mac(
        _mm_permute_mac(tmp.m2, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(up, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_mul_ps(
            _mm_permute_mac(tmp.m2, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_permute_mac(up, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    tmp.m0 = _mm_mul_ps(tmp.m0, _mm_rsqrt_mac(_mm_dp_ps(tmp.m0, tmp.m0, 0b01110111)));
    tmp.m0 = _mm_blend_ps(tmp.m0, _mm_dp_ps(tmp.m0, neg_cam, 0b01111111), 0b1000);

    // up' = -forward x right
    tmp.m1 = _mm_fms_mac(
        _mm_permute_mac(tmp.m0, _MM_SHUFFLE(0, 1, 0, 2)),
        _mm_permute_mac(tmp.m2, _MM_SHUFFLE(0, 0, 2, 1)),
        _mm_mul_ps(
            _mm_permute_mac(tmp.m0, _MM_SHUFFLE(0, 0, 2, 1)),
            _mm_permute_mac(tmp.m2, _MM_SHUFFLE(0, 1, 0, 2))
        )
    );
    tmp.m1 = _mm_blend_ps(tmp.m1, _mm_dp_ps(tmp.m1, neg_cam, 0b01111111), 0b1000);
    tmp.m3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    
    mat4 ret;
    ret.b0 = _mm_shuffle_ps(tmp.m0, tmp.m1, _MM_SHUFFLE(1, 0, 1, 0));
    ret.b1 = _mm_shuffle_ps(tmp.m0, tmp.m1, _MM_SHUFFLE(3, 2, 3, 2));
    ret.b2 = _mm_shuffle_ps(tmp.m2, tmp.m3, _MM_SHUFFLE(1, 0, 1, 0));
    ret.b3 = _mm_shuffle_ps(tmp.m2, tmp.m3, _MM_SHUFFLE(3, 2, 3, 2));
    
    return ret;
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
pure_fn mat4 affine_mat4_right_handed(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#else
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a right-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn mat4 affine_mat4(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#endif
{
    // refer to `rotate_rmat4` for algorithm
    float sin = sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 cos_vec = _mm_mul_ps(
        _mm_set_ps(1.0f, sc_z, sc_y, sc_x),
        _mm_fma_mac(
            _mm_mul_ps(axis, axis),
            _mm_set_ps1(neg_cos),
            _mm_set_ps1(cosf(angle))
        )
    );
    __m128 sin_vec = _mm_mul_ps(axis, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(axis, _MM_SHUFFLE(0, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(axis, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );

    mat4 ret;
    ret.b3 = _mm_mul_ps(
        _mm_set_ps(sc_z, sc_z, sc_y, sc_x),
        _mm_addsub_ps(
            _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 1, 2, 2))
        )
    ); // 2+ 2- 1- 0+
    sin_vec = _mm_mul_ps(
        _mm_set_ps(0.0f, sc_y, sc_x, 0.0f),
        _mm_addsub_ps(
            _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 0, 1, 0)),
            _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 0, 1, 0))
        )
    ); // - 1- 0+ -
    ret.b0 = _mm_shuffle_ps(cos_vec, ret.b3, _MM_SHUFFLE(1, 0, 1, 0)); // c0 c1 2+ 2-
    ret.b0 = _mm_permute_mac(ret.b0, _MM_SHUFFLE(1, 3, 2, 0));

    ret.b1 = _mm_shuffle_ps(translation, sin_vec, _MM_SHUFFLE(2, 1, 1, 0)); // t0 t0 1+ 0-
    ret.b1 = _mm_permute_mac(ret.b1, _MM_SHUFFLE(1, 3, 0, 2));

    ret.b2 = _mm_shuffle_ps(ret.b3, _mm_setzero_ps(), _MM_SHUFFLE(0, 0, 3, 2));

    ret.b3 = _mm_set_ps(1.0f, 0.0f, _mm_extractf_ps(translation, 3), _mm_extractf_ps(cos_vec, 2));
    
    
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
pure_fn mat4 affine_mat4_left_handed(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#else
/// @brief Creates an affine transformation, made by first scaling along the axes, then rotation about an axis in a left-handed system, then translation
/// @param translation The vector to translate by.
/// @param axis The axis to rotate about.
/// @param angle The angle to rotate by.
/// @param sc_x The scaling factor for the x-axis.
/// @param sc_y The scaling factor for the y-axis.
/// @param sc_z The scaling factor for the z-axis.
pure_fn mat4 affine_mat4(const __m128 translation, const __m128 axis, const float angle, const float sc_x, const float sc_y, const float sc_z)
#endif
{
    // refer to `rotate_rmat4` for algorithm
    float sin = -sinf(angle);
    float neg_cos;
    #ifdef VECM_FAST_MATH
        neg_cos = 1.0f - cosf(angle);
    #else
        float half_sin = sinf(angle*0.5f); // 1-cosA = 2sin^2(A/2) for numerical stability
        neg_cos = 2*half_sin*half_sin;
    #endif
    __m128 cos_vec = _mm_mul_ps(
        _mm_set_ps(1.0f, sc_z, sc_y, sc_x),
        _mm_fma_mac(
            _mm_mul_ps(axis, axis),
            _mm_set_ps1(neg_cos),
            _mm_set_ps1(cosf(angle))
        )
    );
    __m128 sin_vec = _mm_mul_ps(axis, _mm_set_ps1(sin));
    __m128 neg_cos_vec = _mm_mul_ps(
        _mm_permute_mac(axis, _MM_SHUFFLE(0, 0, 0, 1)),
        _mm_mul_ps(
            _mm_permute_mac(axis, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_set_ps1(neg_cos)
        )
    );

    mat4 ret;
    ret.b3 = _mm_mul_ps(
        _mm_set_ps(sc_z, sc_z, sc_y, sc_x),
        _mm_addsub_ps(
            _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 1, 2, 2)),
            _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 1, 2, 2))
        )
    ); // 2+ 2- 1- 0+
    sin_vec = _mm_mul_ps(
        _mm_set_ps(0.0f, sc_y, sc_x, 0.0f),
        _mm_addsub_ps(
            _mm_permute_mac(neg_cos_vec, _MM_SHUFFLE(0, 0, 1, 0)),
            _mm_permute_mac(sin_vec, _MM_SHUFFLE(0, 0, 1, 0))
        )
    ); // - 1- 0+ -
    ret.b0 = _mm_shuffle_ps(cos_vec, ret.b3, _MM_SHUFFLE(1, 0, 1, 0)); // c0 c1 2+ 2-
    ret.b0 = _mm_permute_mac(ret.b0, _MM_SHUFFLE(1, 3, 2, 0));

    ret.b1 = _mm_shuffle_ps(translation, sin_vec, _MM_SHUFFLE(2, 1, 1, 0)); // t0 t0 1+ 0-
    ret.b1 = _mm_permute_mac(ret.b1, _MM_SHUFFLE(1, 3, 0, 2));

    ret.b2 = _mm_shuffle_ps(ret.b3, _mm_setzero_ps(), _MM_SHUFFLE(0, 0, 3, 2));

    ret.b3 = _mm_set_ps(1.0f, 0.0f, _mm_extractf_ps(translation, 3), _mm_extractf_ps(cos_vec, 2));
    
    
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
