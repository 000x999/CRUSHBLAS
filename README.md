# CRUSHBLAS
**C++ BLAS-like linear algebra**, **modern encryption** and **lossless compression**

### CURRENT FEATURES 
- Generalized matrix and matrix operations containers using a single contiguous array structure. 
- Threaded Tiled Matrix Multiplications of matrices using AVX256 instructions, speed will depend on the machine, but you can expect 4096x4096 matrices in ~0.51s at 265 GFLOP/s FP32 (CPU Bound).
- Multi-Threaded and Tiled, Matrix Transpose of matrices using AVX256 instructions , speed will depend on the machine, but you can expect 16384x16384 matrices in ~0.889s at 2.25 GB/s FP32 (CPU Bound).

### CURRENTLY WORKING ON 
- VULKAN/OPENGL Compute shader backend as the GPU operation pipeline **==> HIGH PRIORITY**. 
- Cross product and Operation Validity for matrices and matrix operations **==> HIGH PRIORITY**.
- Matrix reductions and reshaping.
- Block kernels, slicing and indexing.
- Eigen Values and vectors. 
- Linear system solvers, LU decomposition, SVD, Cholesky and QR factorization **==> HIGH PRIORITY**.
- Different compression methods and encryption ciphers **==> HIGH PRIORITY**. 

### BENCHMARKS
- MatMul Benchmark:
    ```c++
   void MatMulBenchmark(float A){
      double totalOps = 2.0 * double(A) * double(A) * double(A);
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 
      mat::matrix mat1(A, A);
      mat::matrix mat2(A, A); 
      mat::mat_ops op1(mat1); 
      mat::mat_ops op2(mat2);
      op1.fill_mat();
      op2.fill_mat(); 
      auto start = nanos(); 
      mat::mat_ops op3 = mat::matops::mat_mul(op1,op2); 
      auto end = nanos(); 
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX MatMul: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }

    int main(){
       //First argument is the size of the matrix (Multiple of 8)
       MatMulBenchmark(4096);
    }

    //==========Benchmark output==========:
    137.439 GFLOP
    DEBUG: AVX_MATMUL_STARTED
    AVX MatMul: 0.517075s, GFLOP/S = 265.801


- Transpose Benchmark:
    ```c++
   void TransposeBenchmark(float A){
      double totalOps =  double(A) * double(A);
      double memfactor = 2.0 * A *  A * sizeof(float);
      double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
      std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
      mat::matrix mat1(A, A);
      mat::mat_ops op1(mat1); 
      op1.fill_mat();
      auto start = nanos();
      op1 = op1.transpose();
      auto end = nanos(); 
      double optTime = (end - start) * 1e-9;
      double optmem =  memfactorgb / optTime;
      std::cout << "Transpose: " << optTime
              << "s, GB/S = " << optmem << "\n";
    }

    int main(){
       //First argument is the size of the matrix (Multiple of 8)
       TransposeBenchmark(8192);
    }

    //==========Benchmark output==========:
    67.1089 KB
    Transpose: 0.160259s, GB/S = 3.11995


### EXPECTED FEATURE LIST 
(Please note this is a temporary list, 
I'm still mapping out all the features and their implementations)
- BLAS-level linear algebra operations
- AES-256 and ChaCha20 encryption
- LZ4 and Zstd compression
- Zero-copy encrypted compressed data
- CPU/GPU acceleration support
- Header-only C++20 implementation
- SIMD-optimized kernels
- Sparse matrix compression
- Memory-mapped I/O support
- Thread-safe implementations
