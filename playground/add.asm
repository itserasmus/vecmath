section .text
    global addf      ; Expose the function symbol to the linker
    global ret_mat4  ; Test the most barebones possible function cuz nothing's working :(

addf:
    ; Arguments: a (xmm0), b (xmm1)
    addss xmm0, xmm1 ; Add the two floats in xmm0 and xmm1, store result in xmm0
    ret              ; Return, with result in xmm0


ret_mat4:
    
    ; Allocate space for 4 __m128 (64 bytes) and align stack to 16 bytes
    ; sub rsp, 64                ; Allocate space for 4 __m128 (64 bytes)
    ; and rsp, -16               ; Ensure 16-byte alignment (for SIMD instructions)
    
    movups [rsp], xmm0         ; Store xmm0 in memory
    movups [rsp+16], xmm1      ; Store xmm1 in memory
    movups [rsp+32], xmm2      ; Store xmm2 in memory
    movups [rsp+48], xmm3      ; Store xmm3 in memory

    mov rdi, rsp               ; Return the pointer to the memory (in rdi)
    ret