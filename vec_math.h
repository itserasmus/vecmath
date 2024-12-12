/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This is the main header for the VecMath library.
 * It validates the configuration in `define.h` and imports the correct version(s) of the library.
 */

#ifndef VEC_MATH_H
#define VEC_MATH_H

#include "define.h"

#ifdef VECM_USE_SSE
    #include "sse/vec_math_sse.h"
#elif defined(VECM_USE_AVX)
    #include "avx/vec_math_avx.h"
#elif defined(VECM_USE_AVX512)
    #include "avx512/vec_math_avx512.h"
#endif


#endif