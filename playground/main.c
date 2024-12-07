/**
 * The code and concepts in this folder are experimental. They may not reflect
 * the final implementation or adhere to the quality and performance standards
 * of the main VecMath library.
 * 
 * - This folder serves as a testing ground for new ideas, optimizations, and
 *   approaches before potential integration into the main library.
 * - Features and implementations here are subject to change, removal, or
 *   replacement without notice.
 * - Use at your own risk.
 */
#include <stdio.h>
#include <immintrin.h>
#include <string.h>

#ifndef __clang__
    #define __attribute__(x)
#endif

typedef struct mat4 {
    __m128 m0, m1, m2, m3;
} __attribute__((aligned(64))) mat4;

// Declare the assembly function
extern float addf(float a, float b);
extern mat4 ret_mat4(mat4 a);

void print_mat4(mat4);
mat4 create_mat4(float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float);

int main() {
    mat4 a = create_mat4(
        4, 2, 6, 3,
        6, 3, 7, 8,
        2, 6, 2, 6,
        7, 3, 3, 2);
    mat4 b = create_mat4(
        5, 2, 2, 3,
        3, 7, 6, 7,
        6, 3, 2, 2,
        7, 6, 7, 6);

    mat4 c = ret_mat4(a);

    print_mat4(c);
    return 0;
}


// just for printing
void print_mat4(const mat4 a) {
    _Alignas(16) float a_arr[16];
    memcpy(a_arr, &a, sizeof(mat4));
    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n",
        a_arr[0], a_arr[1], a_arr[2], a_arr[3],
        a_arr[4], a_arr[5], a_arr[6], a_arr[7],
        a_arr[8], a_arr[9], a_arr[10], a_arr[11],
        a_arr[12], a_arr[13], a_arr[14], a_arr[15]
    );
}
mat4 create_mat4(
    const float a, const float b, const float c, const float d,
    const float h, const float i, const float j, const float k,
    const float p, const float q, const float r, const float s,
    const float x, const float y, const float z, const float w) {
    mat4 mat;
    mat.m0 = _mm_set_ps(d, c, b, a);
    mat.m1 = _mm_set_ps(k, j, i, h);
    mat.m2 = _mm_set_ps(s, r, q, p);
    mat.m3 = _mm_set_ps(w, z, y, x);
    return mat;
}