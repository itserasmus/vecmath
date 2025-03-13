/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This file contains macros uesd in the SSE implementation of VecMath. These
 * can be configured by using the options in `define.h`
 */

#include <immintrin.h>
#ifndef MACROS_H_SSE
#define MACROS_H_SSE
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace sse {
#endif
extern "C" {
#endif
#ifdef VECM_USE_FMA
    #define _mm_fma_mac(a, b, c) \
        _mm_fmadd_ps(a, b, c)
    #define _mm_fms_mac(a, b, c) \
        _mm_fmsub_ps(a, b, c)
    #define _mm_fmas_mac(a, b, c) \
        _mm_fmaddsub_ps(a, b, c)
#else
    #define _mm_fma_mac(a, b, c) \
        _mm_add_ps( \
            _mm_mul_ps(a, b), \
            c \
        )
    #define _mm_fms_mac(a, b, c) \
        _mm_sub_ps( \
            _mm_mul_ps(a, b), \
            c \
        )
    #define _mm_fmas_mac(a, b, c) \
        _mm_addsub_ps( \
            _mm_mul_ps(a, b), \
            c \
        )
#endif

#ifdef VECM_EXACT_INV_SQRT
    #define _mm_rsqrt_mac(x) _mm_div_ps(_mm_set_ps1(1.0f), _mm_sqrt_ps(x))
#else
    #define _mm_rsqrt_mac(x) _mm_rsqrt_ps(x)
#endif

#ifdef VECM_EXACT_RECIPROCALS
    #define _mm_rcp_mac(x) _mm_div_ps(_mm_set_ps1(1.0f), x)
#else
    #define _mm_rcp_mac(x) _mm_rcp_ps(x)
#endif

#ifdef VECM_EXACT_DIVISION
    #define _mm_div_mac(x, y) _mm_div_ps(x, y)
#else
    #define _mm_div_mac(x, y) _mm_mul_ps(x, _mm_rcp_ps(y));
#endif

#ifdef VECM_USE_MM_PERMUTE_PS
    #ifndef VECM_SUPRESS_WARNINGS
        #warning "_mm_shuffle_ps is strictly faster than _mm_permute_ps (2024)."
    #endif
    #define _mm_permute_mac(a, m) _mm_permute_ps(a, m)
#else
    #define _mm_permute_mac(a, m) _mm_shuffle_ps(a, a, m)
#endif


#ifdef VECM_ASSERT_DIMENSIONS_COMPATIBILITY
    #define VECM_DIM_ASSERT(cond, err) \
        if(!(cond)) {fprintf(stderr, err);}
#else
    #define VECM_DIM_ASSERT(cond, err)
#endif


#ifndef __cplusplus
    #define reinterpret_int_float(x) ((union { int i; float f; }){ .i = (x) }).f
    #define _mm_extractf_ps(a, x) ((union { int i; float f; }){ .i = _mm_extract_ps(a, x) }).f
#else
    float reinterpret_int_float(int x) {
        return *reinterpret_cast<float*>(&x);
    }
    float _mm_extractf_ps(__m128 a, int x) {
        int tmp = _mm_extract_ps(a, x);
        return *reinterpret_cast<float*>(&tmp);
    }
#endif

#if defined(__GNUC__) || defined(__clang__) ||  defined(__ARMCC_VERSION) ||  defined(__EMSCRIPTEN__)
    #define pure_fn __attribute__((pure))
    #define VECM_ALIGN_16 __attribute__((aligned(16)))
    #define VECM_ALIGN_32 __attribute__((aligned(32)))
    #define VECM_ALIGN_64 __attribute__((aligned(64)))
#elif defined(_MSC_VER) ||  defined(__INTEL_COMPILER) // both use same syntax
    #define pure_fn __declspec(pure)
    #define VECM_ALIGN_16 __declspec(align(16))
    #define VECM_ALIGN_32 __declspec(align(32))
    #define VECM_ALIGN_64 __declspec(align(64))
#else
    #if !defined(VECM_SUPRESS_WARNINGS) && \
        (!defined(VECM_ALIGN_16) || !defined(VECM_ALIGN_32) || !defined(VECM_ALIGN_64))
        #error "Unsupported compiler or missing alignment support. Define VECM_ALIGN_XX manually."
        #define pure_fn
        #define VECM_ALIGN_16
        #define VECM_ALIGN_32
        #define VECM_ALIGN_64
    #endif
#endif

#ifdef __cplusplus
}
}
#ifdef VECM_DOUBLE_NAMESPACE
}
#endif
#endif
#endif