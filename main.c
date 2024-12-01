/**
 * VecMath - SIMD-based Matrix and Vector Library
 * 
 * This file is part of the VecMath project and is licensed under the MIT License.
 * See the LICENSE file in the root of the repository for full license details.
 * 
 * Copyright (c) 2024 Om Patil
 */

/**
 * This file is for testing and profiling of VecMath functions.
 */
#include <stdio.h>
#include <time.h>
#include <Windows.h>

#define VECM_USE_AVX
#define VECM_USE_FMA
// #define VECM_FAST_MATH
#include "vec_math.h"

void pin_thread_to_curr_core();


int main(int argc, char **argv) {
    // bmat4 a = create_bmat4(
    //     4, 1, -2, 1,
    //     2, 3, 1, 1,
    //     -1, 2, 2, 1,
    //     5, -3, 3, 7);
    // volatile bmat4 b = create_bmat4(
    //     3, 5, 1, 1,
    //     7, 1, -1, 4,
    //     5, -1, 4, 8,
    //     6, 7, 2, 3);
    mat4 ord = create_mat4(
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15);

    mat4 a = create_mat4(
         4,  1, -2,  1,
         2,  3,  1,  1,
        -1,  2,  2,  1,
         5, -3,  3,  7);
    volatile mat4 b = create_mat4(
         3,  5,  1,  1,
         7,  1, -1,  4,
         5, -1,  4,  8,
         6,  7,  2,  3);

    vec4 vord = create_vec4(0, 1, 2, 3);
    vec4 v1 = create_vec4(4, 5, 2, 8);
    vec4 v2 = create_vec4(7, 1, 3, 2);

    float k = 69;

    // #define TEST_FUNCTION_A mul_vec4_mat4
    // #define TV_A_1 v1
    // #define TV_A_2 b
    // #define TEST_FUNCTION_B mul_vec4_mat4lol
    // #define TV_B_1 v1
    // #define TV_B_2 b

    // struct timespec start, end;
    // pin_thread_to_curr_core();
    // // warmup loop for jit optimization
    // for (long long int i = 0; i < 100000000; i++) {
    //     k += _mm256_cvtss_f32(trans_mat4(a).m0);
    // }
    // const long long MILLION = 1000000L;
    // const long long BILLION = 1000000000L;
    // const long long TRILLION = 1000000000000L;
    // const long long NUM_ITERATIONS = 1*BILLION;
    // timespec_get(&start, TIME_UTC);
    // for (long long int i = 0; i < 
    // NUM_ITERATIONS/100
    // ; i++) {
    //     // #define TEST_VAL_A TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2) + TEST_FUNCTION_A(TV_A_2)  // 10 iterations
    //     // k += TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A + TEST_VAL_A; // 100 iterations
    //     // #define TEST_VAL_A(TV_A____) TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TEST_FUNCTION_A(TV_A_1, TV_A____)))))))))) // 10 iterations
    //     // k += _mm_cvtss_f32(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TV_A_2)))))))))).b0); // 100 iterations
    //     #define TEST_VAL_A(TV_A____) TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TV_A____, TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2), TV_A_2) // 10 iterations
    //     k += _mm_cvtss_f32(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TV_A_1))))))))))); // 100 iterations
    //     // #define TEST_VAL_A(TV_A____) TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TEST_FUNCTION_A(TV_A____)))))))))) // 10 iterations
    //     // k += _mm_cvtss_f32(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TEST_VAL_A(TV_A_2))))))))))); // 100 iterations
    // }
    // timespec_get(&end, TIME_UTC);
    // double duration = ((double)end.tv_sec*1e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec);
    // printf("time 1  : %i ms\n", (int)(duration/1000000));
    // timespec_get(&start, TIME_UTC);
    // for (long long int i = 0; i < 
    // NUM_ITERATIONS/100
    // ; i++) {
    //     // #define TEST_VAL_B TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2) + TEST_FUNCTION_B(TV_B_2)  // 10 iterations
    //     // k += TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B + TEST_VAL_B; // 100 iterations
    //     // #define TEST_VAL_B(TV_B____) TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TEST_FUNCTION_B(TV_B_1, TV_B____)))))))))) // 10 iterations
    //     // k += _mm_cvtss_f32(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TV_B_2)))))))))).m0); // 100 iterations
    //     #define TEST_VAL_B(TV_B____) TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TV_B____, TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2), TV_B_2) // 10 iterations
    //     k += _mm_cvtss_f32(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TV_B_1))))))))))); // 100 iterations        
    //     // #define TEST_VAL_B(TV_B____) TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TEST_FUNCTION_B(TV_B____)))))))))) // 10 iterations
    //     // k += _mm_cvtss_f32(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TEST_VAL_B(TV_B_2)))))))))).m0); // 100 iterations   
    // }
    // timespec_get(&end, TIME_UTC);
    // duration = ((double)end.tv_sec*1e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec);
    // printf("time 2  : %i ms\n", (int)(duration/1000000));


    // print_rmat4(inv_rmat4(look_at_rmat4(
    //     create_vec3(1, 1, 5), // camera
    //     create_vec3(0, 0, 0), // target
    //     create_vec3(0, 1, 0)  // up
    // )));
    // print_vec4(rotate_vec3(create_vec3(1, 1, 5), norm_vec3(create_vec3(4, 2, 6)), 3.14159*0.25));
    printf("%f\n", det_mat4(a));
    // print_vec4(mul_vec4_mat4(v1, a));
    // print_mat4(trans_mat4(ord));

    if(k == INFINITY) {printf("%f", k);}
    
    return 0;
}

void pin_thread_to_curr_core() {
    DWORD current_core = GetCurrentProcessorNumber();
    HANDLE current_thread = GetCurrentThread();  // Get handle to the current thread
    DWORD_PTR affinity_mask = 1ULL << current_core;  // Create a bitmask for the desired core

    // Set the thread affinity
    DWORD_PTR result = SetThreadAffinityMask(current_thread, affinity_mask);
    if (result == 0) {
        printf("Failed to set thread affinity. Error code: %lu\n", GetLastError());
    } else {
        printf("Thread pinned to core %lu successfully.\n", current_core);
    }
}