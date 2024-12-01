/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Main file for AVX implementation of VecMath. Recommended to import through
 * `vec_math.h` or import `define.h` before this file.
 */

#ifndef VEC_MATH_H_AVX
#define VEC_MATH_H_AVX

#ifdef __cplusplus
#include <cstdio>
#endif

#include <math.h>
#include <immintrin.h>
#include <string.h>

#include "macros_avx.h"

#include "common_avx.h"
#include "constructors_avx.h"

#include "vec2x2_avx.h"
#include "vec3x3_avx.h"
#include "vec4x4_avx.h"

#include "mat2x2_avx.h"
#include "mat3x3_avx.h"
#include "mat4x4_avx.h"


#include "block_mat4x4_avx.h"



#endif
