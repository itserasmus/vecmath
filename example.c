/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This is an example project using VecMath.
 */

#define VECM_USE_SSE        // Use sse functions
#define VECM_FAST_MATH      // Uses approximations for division and reciplrocals
#define VECM_USE_FMA        // Uses fma where it can. Make sure to use the flag `-mfma`
#include "vec_math.h"       // Include vecmath with the defined options

int main() {
    /* Example 1: Creating and using vectors */
    vec4 v1 = create_vec4(2.0f, 5.0f, 3.0f, 2.0f);      // Using the inbuilt function
    vec4 v2 = _mm_setr_ps(4.0f, 6.0f, 2.0f, 3.0f);      // Using immintrin natives
    vec4 sum = add_vec4(v1, v2);                        // Sum
    print_vec4(sum);                                    // Print sum

    vec3 v3 = create_vec3(2.0f, 4.0f, 3.0f);            // Create a 3-vector
    vec3 v4 = _mm_setr_ps(5.0f, 4.0f, 7.0f, 0.0f);      // Use immintrin natives. The last element is undefined, but is usually 0
    vec3 cross = cross_vec3(v3, v4);                    // Cross product
    print_vec3(cross);                                  // Print cross
    


    /* Example 2: Creating and using matrices */
    mat3 m1 = create_mat3(3.0f, 5.0f, 4.0f,             // Using the inbuilt function.
                          4.0f, 6.0f, 7.0f,
                          2.0f, 1.0f, 8.0f);
    mat3 m2;                                            // Using immintrin natives
    m2.m0 = _mm_setr_ps(4.0f, 6.0f, 7.0f, 0.0f);
    m2.m1 = _mm_setr_ps(6.0f, 8.0f, 8.0f, 0.0f);
    m2.m2 = _mm_setr_ps(3.0f, 3.0f, 4.0f, 0.0f);
    mat3 prod1 = mul_mat3(m1, m2);                      // Multiply the matrices
    print_mat3(prod1);                                  // Print product

    vec3 prod2 = mul_vec3_mat3(v3, m1);                 // v3 * m1, pre multiplication by a vector
    vec3 prod2 = mul_mat3_vec3(m1, v3);                 // m1 * v3, post multiplication by a vector



    /* Example 3: Basic graphics functions */
    rmat4 lookat = look_at_rmat4(v3, v4, create_vec3(0.0f, 1.0f, 0.0f));    // Create a lookat matrix (camera, target, up)
    rmat4 reflection = reflect_rmat4(v3);                                   // Create a reflection across the plane normal to v3



    /* Example 4: Unsafe operations */
    cross = cross_vec3(v1, v2);
    // Notice that this operation is valid, despite `v1` and `v2` being of
    // type `vec4` and not `vec3`. This is because both the `vec3` and
    // `vec4` class are stored as `__m128`. This is an unsafe but legal
    // operation. It also works with `vec2` and `mat2`.

    // While casting vec3 to vec4, note that the last element of `vec3`
    // is undefined, not zero. Undefined values can cause unpredictable results
    // in calculations that depend on the extra component. To explicitly set
    // the last element to zero:
    vec4 v5 = _mm_blend_ps(v3, _mm_setzero_ps(), 0b1000);           // Takes first three elements from `v3` and last one from `_mm_setzero_ps()`
    
    // mat4 m3 = m1;
    // If uncommented, this line would cause an error. This is because `mat3`
    // and `mat4` are stored using different structs, unline `vec3` and `vec4`.
    // They are incompatible, and implicit size conversions are not supported.
}