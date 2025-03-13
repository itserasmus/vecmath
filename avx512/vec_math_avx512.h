/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * Main file for AVX512 implementation of VecMath. Recommended to import through
 * `vec_math.h` or import `define.h` before this file.
 */

#ifndef VEC_MATH_H_AVX512
#define VEC_MATH_H_AVX512

#ifdef __cplusplus
#include <cstdio>
#endif

#include <math.h>
#include <immintrin.h>
#include <string.h>

#include "macros_avx512.h"

#include "common_avx512.h"
#include "constructors_avx512.h"

#include "vec2_avx512.h"
#include "vec3_avx512.h"
#include "vec4_avx512.h"

#include "mat2x2_avx512.h"


#endif
