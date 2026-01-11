# CRUSHBLAS

A high-performance BLAS library written in C++ from scratch with **NO** external dependencies and hand-tuned AVX2/AVX-512 kernels.
## Performance Summary

| Benchmark | Configuration | Result |
|-----------|---------------|--------|
| GEMM (FP32) | 4096x4096 | **~825 GFLOP/s** sustained |
| GEMM (FP32) | 8192x8192 | **~825 GFLOP/s** sustained |
| GEMM (FP32) | 16384x16384 | **~825 GFLOP/s** sustained |
| Softmax | row-wise | Vectorized AVX implementation |

*Run `gemm_test(4096)` in the test bed to reproduce!*

## Features

### Level-3 BLAS Operations
- **GEMM**: Single-precision general matrix multiplication (C = α×A×B + β×C)
- **Transpose-aware**: Native support for transposed operands (N, T flags)
- **Softmax**: Row-wise softmax with numerical stability

### Microkernel Architecture
- **6x16 register-blocked kernel**: 12 YMM/ZMM accumulators for maximum FMA throughput
- **4x8 fallback kernel**: For edge cases and smaller tiles
- **Edge handling**: Scalar fallback for non-aligned boundaries

### Optimization Techniques
- **Cache blocking**: 256x256x256 tiles tuned for L2/L3 cache hierarchy
- **Matrix packing**: Contiguous memory layout for sequential cache line access
- **Register blocking**: Accumulate in registers across entire K-loop, store once
- **Software prefetching**: Next cache line prefetch in inner loop
- **Loop unrolling**: 4x unroll in K dimension for instruction-level parallelism

### Parallelization
- **OpenMP**: `collapse(2)` scheduling across M and N tile dimensions
- **Thread pool**: Custom thread pool implementation for fine-grained control
- **Dynamic scheduling**: Load balancing across heterogeneous workloads

## Building

### Prerequisites
- C++23 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- CMake 3.20+
- Ninja (Linux) or MinGW (Windows)
- OpenMP (optional, for parallelization)
- CPU with AVX2 support (AVX-512 recommended for best performance)

### Linux
```bash
cmake --preset linux-release
./build.sh linux-release linux-release-build ON OFF
./run.sh
```

### Windows
```bash
cmake --preset mingw-release
./build.bat mingw-release mingw-release-build 1 0
./run.bat
```

### Build Script Arguments
```
./build.sh <preset> <build-preset> <AVX> <DEBUG>
```
| Argument | Values | Description |
|----------|--------|-------------|
| `AVX` | ON/OFF (1/0 on Windows) | Enable AVX2/AVX-512 SIMD |
| `DEBUG` | ON/OFF (1/0 on Windows) | Enable debug assertions and logging |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_AVX256` | ON | Enable AVX2/AVX-512 intrinsics |
| `DEBUG_BLAS` | OFF | Enable debug prints and assertions |
| `USE_VULKAN` | OFF | Enable Vulkan compute backend (WIP) |
| `USE_OPENGL` | OFF | Enable OpenGL compute backend (WIP) |

## Usage

### Header
```cpp
#include "BLAS/level3/level3.hpp"
```

### GEMM (Matrix Multiplication)

```cpp
#include "BLAS/level3/level3.hpp"

// Define matrix views
level3::mat_ops_view A = {
    .row_view = M,
    .col_view = K,
    .leading_dimension = K,
    .data_view = a_data
};

level3::mat_ops_view B = {
    .row_view = K,
    .col_view = N,
    .leading_dimension = N,
    .data_view = b_data
};

level3::mat_ops_view C = {
    .row_view = M,
    .col_view = N,
    .leading_dimension = N,
    .data_view = c_data
};

// C = alpha * A @ B + beta * C
level3::blas::crush_gemm(
    level3::transpose_gemm::no_transpose,  // A: no transpose
    level3::transpose_gemm::no_transpose,  // B: no transpose
    A, B,
    1.0f,  // alpha
    0.0f,  // beta
    C
);
```

### Transposed Operations

```cpp
// C = alpha * A^T @ B + beta * C
level3::blas::crush_gemm(
    level3::transpose_gemm::transpose,     // A: transposed
    level3::transpose_gemm::no_transpose,  // B: no transpose
    A, B,
    1.0f, 0.0f, C
);

// C = alpha * A @ B^T + beta * C
level3::blas::crush_gemm(
    level3::transpose_gemm::no_transpose,  // A: no transpose
    level3::transpose_gemm::transpose,     // B: transposed
    A, B,
    1.0f, 0.0f, C
);
```

### Softmax

```cpp
#include "BLAS/level3/level3.hpp"

level3::mat_ops_view input = {
    .row_view = batch_size,
    .col_view = num_classes,
    .leading_dimension = num_classes,
    .data_view = logits
};

// In-place row-wise softmax
level3::mat_ops_view output = level3::blas::softmax(input);
```

### Matrix View Indexing

```cpp
level3::mat_ops_view mat = { /* ... */ };

// Element access
float val = mat(i, j);      // Using operator()
float val = mat[i][j];      // Using operator[] with map_view

// Row iteration
for (size_t i = 0; i < mat.row_view; ++i) {
    for (size_t j = 0; j < mat.col_view; ++j) {
        mat[i][j] = /* ... */;
    }
}
```

## Benchmarks

All benchmarks performed on AMD Ryzen 9 9950X3D and an Intel i9-9900K with OpenMP parallelization enabled.

### GEMM Performance (FP32)

| Matrix Size | GFLOP | Time | Throughput | % Peak |
|-------------|-------|------|------------|--------|
| 1024 x 1024 | 2.1 | 2.6ms | ~820 GFLOP/s | ~23% |
| 4096 x 4096 | 137.4 | 0.167s | **~825 GFLOP/s** | ~23% |
| 8192 x 8192 | 1099.5 | 1.33s | **~825 GFLOP/s** | ~23% |
| 16384 x 16384 | 8796.1 | 10.66s | **~825 GFLOP/s** | ~23% |

*% Peak calculated against theoretical max of ~3.5 TFLOP/s (16 cores, AVX-512, FMA)*

### Comparison with Production Libraries

| Library | 4096x4096 GEMM | Notes |
|---------|-----------------|-------|
| **CRUSHBLAS** | 825 GFLOP/s | From-scratch |
| OpenBLAS | ~3.5 TFLOP/s | Highly optimized, assembly kernels |
| Intel MKL | ~3.8 TFLOP/s | Vendor-optimized |

CRUSHBLAS achieves ~23% of theoretical peak, the kernels still have a long way to go! The gap with production libraries comes from:
- Assembly-level optimization in OpenBLAS/MKL
- Architecture-specific tuning and dispatch
- More aggressive prefetching strategies
- Kernel specialization for specific matrix sizes

## API Reference

### Core Functions

```cpp
// Primary GEMM interface (recommended)
void level3::blas::crush_gemm(
    level3::transpose_gemm transpose_left,   // no_transpose or transpose
    level3::transpose_gemm transpose_right,  // no_transpose or transpose
    const level3::mat_ops_view &A,
    const level3::mat_ops_view &B,
    float alpha,
    float beta,
    level3::mat_ops_view &C
);

// Softmax
level3::mat_ops_view blas::softmax(mat_ops_view &input);

// Low-level GEMM variants
void level3::blas::gemm_avx256(...);     // Single-threaded AVX2
void level3::blas::gemm_avx256_mt(...);  // Multi-threaded AVX2
```

### Data Structures

```cpp
namespace level3 {
// Transpose flag
enum class transpose_gemm {
    no_transpose,
    transpose
};

// Matrix view (non-owning)
struct mat_ops_view {
    size_t row_view;           // Number of rows
    size_t col_view;           // Number of columns
    size_t leading_dimension;  // Stride between rows
    float *data_view;          // Pointer to data
    
    float &operator()(size_t i, size_t j);
    map_view operator[](size_t row);
};

// Thread pool for custom parallelization
class gemm_thread_pool {
    static gemm_thread_pool &get_instance();
    void init_pool(size_t num_threads);
    void enqueue_task(std::function<void()> task);
    void wait_for_all();
    void shutdown_pool();
};
}
```

## Project Structure

```
CRUSHBLAS/
├── core/
│   ├── BLAS/
│   │   └── level3/
│   │       └── level3.hpp         # Public API header
│   ├── include/
│   │   └── crush_defines.h        # Macros and constants
│   └── impl/
│       └── blas_impl/
│           └── level3/
│               └── level3.cpp     # Implementation
├── test_bed/
│   └── src/                       # Tests and benchmarks
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh / build.bat
└── run.sh / run.bat
```

## Roadmap

### In Progress
- [ ] FP16 (half-precision) GEMM kernels
- [ ] INT8 quantized GEMM for inference
- [ ] Vulkan compute shader backend
- [ ] Assembly-level optimization
- [ ] More aggressive prefetching strategies
- [ ] Kernel specialization for specific matrix sizes


### Planned
- [ ] Level-1 BLAS (vector operations)
- [ ] Level-2 BLAS (matrix-vector operations)
- [ ] Batched GEMM for transformer workloads
- [ ] ARM NEON backend for mobile/Apple Silicon
- [ ] JIT/Compiler for automatic CPU/GPU code generation
- [ ] Auto-tuning for optimal tile sizes per architecture

## Technical Notes

### Why 6x16 Microkernel?

The 6x16 tile size is chosen to maximize register utilization on AVX2/AVX-512:
- 6 rows × 2 YMM registers (16 floats) = 12 accumulator registers
- Leaves 4 registers for A broadcast, B loads, and temporaries
- Fits comfortably in the 16 YMM register file

### Cache Blocking Strategy

```
L1 Cache (32KB):  Microkernel working set (~6KB)
L2 Cache (512KB): K-blocking panel (~256KB)
L3 Cache (32MB):  Full tile reuse across M/N blocks
```

The 256x256x256 block size ensures:
- Packed A panel fits in L2
- Packed B panel streams from L3
- C tile accumulates in registers

### Memory Layout

```
Standard (row-major):     Packed (microkernel-friendly):
A[i][k] = A[i * K + k]    A_pack[i * K + k] (contiguous rows)
B[k][j] = B[k * N + j]    B_pack[k * NR + j] (contiguous panels)
```

Packing eliminates TLB misses and ensures sequential cache line access in the inner loop.

## License

MIT

## Acknowledgments

- Inspired by BLIS, OpenBLAS, and GotoBLAS2 microkernel designs
- Reference: Goto & Van De Geijn, "Anatomy of High-Performance Matrix Multiplication"
