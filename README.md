### VecMath
VecMath is a highly optimized SIMD-based C library, interoperable with C++ for matrix and vector mathematics. It includes support for matrices and vectors up to size 4, with planned support for arbitrarily large matrices. This library also include basic graphics features, with more advanced ones planned for the future.

## Features
- Block Matrix: The default matrix for 4x4 (and beyond, in the future). Block matrices offer improved performance over row-major matrices.
- Optimized with SIMD: Uses SIMD intrinsics for maximum speed. Currently supports SSE, with almost full AVX support, and planned support for AVX512.
- Graphics Extension: Adds basic graphics functions.
- Optimized yet Abstracted: VecMath is highly optimized using SIMD intrinsics and block matrices, while maintaining sufficient abstraction for ease of use.

## Planned Features
- AVX512 implementation
- mxn matrices
- Advanced graphics functions
- Advanced linear algebra functions

## Getting Started
# Requirements
- C/C++ compiler supporting SIMD (GCC/G++, Clang, MSVC)
- Optionally, Git for version control
# Usage
- VecMath is a header-only library and does not need building
- Clone the repositry
- Import `path/to/vecmath/vec_math.h`
- Define VecMath flags. For a full list, see the comment at the top of [`define.h`](./define.h)
- Compile your program. Make sure to use `-msse`, `-mavx` or `-mavx512` to enable SIMD intrinsics
  ```
  gcc -O3 -mavx main.c -o main.exe
  ```
- For an example, refer to [`example.c`](./example.c)

## Contributing
Contributions are welcome. Feel free to fork the repo and submit pull requests for
- Bug Fixes
- Performance Optimizations
- New Features

## License
The project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.