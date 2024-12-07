/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This file contains macros uesd in the AVX512 implementation of VecMath. These
 * can be configured by using the options in `define.h`
 */

#include <immintrin.h>
#ifndef MACROS_H_AVX512
#define MACROS_H_AVX512
#ifdef __cplusplus
namespace vecm {
#ifdef VECM_DOUBLE_NAMESPACE
namespace avx512 {
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
    #define _mm256_fma_mac(a, b, c) \
        _mm256_fmadd_ps(a, b, c)
    #define _mm256_fms_mac(a, b, c) \
        _mm256_fmsub_ps(a, b, c)
    #define _mm256_fmas_mac(a, b, c) \
        _mm256_fmaddsub_ps(a, b, c)
    #define _mm512_fma_mac(a, b, c) \
        _mm512_fmadd_ps(a, b, c)
    #define _mm512_fms_mac(a, b, c) \
        _mm512_fmsub_ps(a, b, c)
    #define _mm512_fmas_mac(a, b, c) \
        _mm512_fmaddsub_ps(a, b, c)
#else
    #define _mm_fma_mac(a, b, c) \
        _mm_add_ps(_mm_mul_ps(a, b), c)
    #define _mm_fms_mac(a, b, c) \
        _mm_sub_ps(_mm_mul_ps(a, b), c)
    #define _mm_fmas_mac(a, b, c) \
        _mm_addsub_ps(_mm_mul_ps(a, b), c)
    #define _mm256_fma_mac(a, b, c) \
        _mm256_add_ps(_mm256_mul_ps(a, b), c)
    #define _mm256_fms_mac(a, b, c) \
        _mm256_sub_ps(_mm256_mul_ps(a, b), c)
    #define _mm256_fmas_mac(a, b, c) \
        _mm256_addsub_ps(_mm256_mul_ps(a, b), c)
    #define _mm512_fma_mac(a, b, c) \
        _mm512_add_ps(_mm512_mul_ps(a, b), c)
    #define _mm512_fms_mac(a, b, c) \
        _mm512_sub_ps(_mm512_mul_ps(a, b), c)
    // for some reason, _mm512 does not have an _mm512_addsub_ps,
    // so use a xor mask to negate alternate elements
    #define _mm512_fmas_mac(a, b, c) \
        _mm512_add_ps(_mm512_mul_ps(a, b), _mm512_xor_ps(c, _mm512_set4_ps(0.0f,-0.0f,0.0f,-0.0f)))
#endif

#ifdef VECM_EXACT_INV_SQRT
    #define _mm_rsqrt_mac(x) _mm_div_ps(_mm_set_ps1(1.0f), _mm_sqrt_ps(x))
    #define _mm256_rsqrt_mac(x) _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(x))
    #define _mm512_rsqrt_mac(x) _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_sqrt_ps(x))
#else
    #define _mm_rsqrt_mac(x) _mm_rsqrt_ps(x)
    #define _mm256_rsqrt_mac(x) _mm256_rsqrt_ps(x)
    #define _mm512_rsqrt_mac(x) _mm512_rsqrt14_ps(x)
#endif

#ifdef VECM_EXACT_RECIPROCALS
    #define _mm_rcp_mac(x) _mm_div_ps(_mm_set_ps1(1.0f), x)
    #define _mm256_rcp_mac(x) _mm256_div_ps(_mm256_set1_ps(1.0f), x)
    #define _mm512_rcp_mac(x) _mm512_div_ps(_mm512_set1_ps(1.0f), x)
#else
    #define _mm_rcp_mac(x) _mm_rcp_ps(x)
    #define _mm256_rcp_mac(x) _mm256_rcp_ps(x)
    #define _mm512_rcp_mac(x) _mm512_rcp14_ps(x)
#endif

#ifdef VECM_EXACT_DIVISION
    #define _mm_div_mac(x, y) _mm_div_ps(x, y)
    #define _mm256_div_mac(x, y) _mm256_div_ps(x, y)
    #define _mm512_div_mac(x, y) _mm512_div_ps(x, y)
#else
    #define _mm_div_mac(x, y) _mm_mul_ps(x, _mm_rcp_ps(y));
    #define _mm256_div_mac(x, y) _mm256_mul_ps(x, _mm256_rcp_ps(y));
    #define _mm512_div_mac(x, y) _mm512_mul_ps(x, _mm512_rcp14_ps(y));
#endif

#ifdef VECM_USE_MM_PERMUTE_PS
    #ifndef VECM_SUPRESS_WARNINGS
        #warning "_mm_shuffle_ps is strictly faster than _mm_permute_ps (2024)."
    #endif
    #define _mm_permute_mac(a, m) _mm_permute_ps(a, m)
    #define _mm256_permute_mac(a, m) _mm256_permute_ps(a, m)
    #define _mm512_permute_mac(a, m) _mm512_permute_ps(a, m)
#else
    #define _mm_permute_mac(a, m) _mm_shuffle_ps(a, a, m)
    #define _mm256_permute_mac(a, m) _mm256_shuffle_ps(a, a, m)
    #define _mm512_permute_mac(a, m) _mm512_shuffle_ps(a, a, m)
#endif


#define pure_fn __attribute__((pure))


#ifndef __cplusplus
    #define reinterpret_int_float(x) ((union { int i; float f; }){ .i = (x) }).f
    #define _mm_extractf_ps(a, x) ((union { int i; float f; }){ .i = _mm_extract_ps((a), (x)) }).f
#else
    float reinterpret_int_float(int x) {
        return *reinterpret_cast<float*>(&x);
    }
    float _mm_extractf_ps(__m128 a, int x) {
        int tmp = _mm_extract_ps(a, x);
        return *reinterpret_cast<float*>(&tmp);
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