/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Main file for SSE implementation of VecMath. Recommended to import through
 * `vec_math.h` or import `define.h` before this file.
 */

#ifndef VEC_MATH_H_SSE
#define VEC_MATH_H_SSE

#ifdef __cplusplus
#include <cstdio>
#endif

#include <math.h>
#include <immintrin.h>
#include <string.h>

#include "macros_sse.h"

#include "common_sse.h"
#include "constructors_sse.h"

#include "vec2x2_sse.h"
#include "vec3x3_sse.h"
#include "vec4x4_sse.h"

#include "mat2x2_sse.h"
#include "mat3x3_sse.h"
#include "mat4x4_sse.h"
#include "mat4x4_row_sse.h"

#include "mat4x4_sse_ext.h"
#include "mat4x4_row_sse_ext.h"

#endif
