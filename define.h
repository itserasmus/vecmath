/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This file validates the configuration for VecMath.
 * Below is a list of all VecMath options:
 * - VECM_APPROX_DIVISION: Uses approximate but faster division. Overrides VECM_EXACT_MATH
 * - VECM_APPROX_INV_SQRT: Uses approximate but faster square roots. Overrides VECM_EXACT_MATH
 * - VECM_APPROX_RECIPROCALS: Uses approximate but faster reciprocals. Overrides VECM_EXACT_MATH
 * - VECM_DOUBLE_NAMESPACE: C++ only. Creates a sub-namespace under `vecm` for sse, avx and avx512 when importing two or more of them.
 * - VECM_EXACT_DIVISION: Uses exact but slower division. Overrides VECM_FAST_MATH
 * - VECM_EXACT_INV_SQRT: Uses exact but slower inverse square roots. Overrides VECM_FAST_MATH
 * - VECM_EXACT_MATH: Uses exact but slow divisions and reciprocals
 * - VECM_EXACT_RECIPROCALS: Uses exact but slower reciprocals. Overrides VECM_FAST_MATH
 * - VECM_FAST_MATH: Uses approximate but fast divisions and reciprocals
 * - VECM_HANDEDNESS_NAMING: Names functions based on handedness when importing both left and right handed functions
 * - VECM_LEFT_HANDED: Uses a left handed coordinate system (only affects pseudovectors)
 * - VECM_RIGHT_HANDED: Uses a right handed coordinate system (only affects pseudovectors)
 * - VECM_SUPRESS_WARNINGS: Supresses compile-time warnings
 * - VECM_NO_FMA: Prevents usage of FMA instructions
 * - VECM_USE_AVX: Uses the AVX based VecMath implementation
 * - VECM_USE_AVX512: Uses the AVX512 based VecMath implementation (not supported yet)
 * - VECM_USE_FMA: Uses FMA instuctions. Requires `-mfma` flag
 * - VECM_USE_SSE: Uses the SSE based VecMath implementation
 */

#ifndef VECM_SETTINGS_H
#define VECM_SETTINGS_H

#if !defined(VECM_EXACT_MATH) && !defined(VECM_FAST_MATH)
    #ifdef  __FAST_MATH__
        #define VECM_FAST_MATH
    #else
        #define VECM_EXACT_MATH
    #endif
#endif

#if !defined(VECM_USE_SSE) && !defined(VECM_USE_AVX) && !defined(VECM_USE_AVX512)
    #ifdef __AVX512__
    #define VECM_USE_AVX512
    #elif defined(__AVX__)
    #define VECM_USE_AVX
    #else
    #define VECM_USE_SSE
    #endif
#endif

#if !defined(VECM_EXACT_RECIPROCALS) && !defined(VECM_APPROX_RECIPROCALS)
    #ifdef VECM_FAST_MATH
        #define VECM_APPROX_RECIPROCALS
    #else
        #define VECM_EXACT_RECIPROCALS
    #endif
#endif
#if !defined(VECM_EXACT_DIVISION) && !defined(VECM_APPROX_DIVISION)
    #ifdef VECM_FAST_MATH
        #define VECM_APPROX_DIVISION
    #else
        #define VECM_EXACT_DIVISION
    #endif
#endif
#if !defined(VECM_EXACT_INV_SQRT) && !defined(VECM_APPROX_INV_SQRT)
    #ifdef VECM_FAST_MATH
        #define VECM_APPROX_INV_SQRT
    #else
        #define VECM_EXACT_INV_SQRT
    #endif
#endif

#if defined(__FMA__) && !defined(VECM_NO_FMA) && !defined(VECM_USE_FMA)
    #define VECM_USE_FMA
#endif

#if defined(VECM_RIGHT_HANDED) && defined(VECM_LEFT_HANDED)
    #define VECM_HANDEDNESS_NAMING
#endif

#if !defined(VECM_RIGHT_HANDED) && !defined(VECM_LEFT_HANDED)
    #define VECM_RIGHT_HANDED
#endif

#if (defined(VECM_USE_SSE) + defined(VECM_USE_AVX) + defined(VECM_USE_AVX512)) > 1
    #ifdef __cplusplus
        #define VECM_DOUBLE_NAMESPACE
    #else
        #error "Can only use one of SSE, AVX or AVX512 implementations in C. Use only one implementation or use C++"
    #endif
#endif


#endif