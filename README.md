# CRUSHBLAS
**C++ BLAS**, **modern encryption** and **lossless compression**

### CURRENT FEATURES 
- Generalized Matrix API and Matrix operations container using a single contiguous array structure. 
- Threaded Tiled Matrix Multiplications of matrices using AVX256 instructions, speed will depend on the machine, but one can expect 4096x4096 matrices in ~0.51s at 265 GFLOP/s FP32 (CPU Bound).
- Multi-Threaded and Tiled, Matrix Transpose of matrices using AVX256 instructions , speed will depend on the machine, but one can expect 16384x16384 matrices in ~0.889s at 2.25 GB/s FP32 (CPU Bound).
- Level3 GEMM kernel using the custom Matrix API as well as AVX256 instruction acceleration.
- Generalized Tensor API and Tensor operations container through the custom Matrix API.
- Contracted TensorMul using the level3 GEMM kernel with AVX256 acceleration, speed will depend on the machine and the Tensor structure, but one can expect a Contracted TensorMul with,  ```Tensor(1,15,4096)```, that is, 1 batch, 15 slices of 4096x4096 matrices to be computed in ~8.1893s at 251.741 GFLOP/s FP32 (CPU Bound).
- Batched TensorMul using the level3 GEMM kernel with AVX256 acceleration, speed will depend on the machine and the Tensor structure, but one can expect a Batched TensorMul with,  ```Tensor(1,15,4096)```, that is, 1 batch, 15 slices of 4096x4096 matrices to be computed in ~8.1893s at 237.869 GFLOP/s FP32 (CPU Bound).

### CURRENTLY WORKING ON 
- VULKAN/OPENGL Compute shader backend as the GPU operation pipeline **==> HIGH PRIORITY**. 
- Cross product and Operation Validity for matrices and matrix operations **==> HIGH PRIORITY**.
- Matrix reductions and reshaping.
- Block kernels, slicing and indexing.
- Eigen Values and vectors. 
- Linear system solvers, LU decomposition, SVD, Cholesky and QR factorization **==> HIGH PRIORITY**.
- Different compression methods and encryption ciphers **==> HIGH PRIORITY**. 

### BENCHMARKS
- Contracted TensorMul Benchmark:
    ```c++
    void contracted_tensor_mul_benchmark(size_t batches, size_t slices, size_t matrix_size){
      tens::tensor tensor_a(dims,rank,matrix_size);
      tens::tensor tensor_b(dims,rank,matrix_size);
      tens::tensor_ops tensor_op_a(tensor_a);
      tens::tensor_ops tensor_op_b(tensor_b); 
      tens::tensor_ops::fill_tensor(tensor_op_a);
      tens::tensor_ops::fill_tensor(tensor_op_b);
      std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
      std::cout << "Tensor batches: " << tensor_a.m_batches << std::endl; 
      std::cout << "Tensor slices: " << tensor_a.m_slices << std::endl;
      double totalOps = tensor_a.m_slices * (2 * double(matrix_size) * double(matrix_size) * double(matrix_size));
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl;
      auto start = nanos();
      mat::mat_ops C_mat = tens::tensor_ops::contract_tensor_mul(tensor_op_a, tensor_op_b);
      auto end = nanos();
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX CONTRACTED TENSOR MUL: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }

    int main(){
        //First argument is the total number of tensor batches
        //The second argument is the total number of tensor slices
        //The third argument is the size of the matrices in each tensor slice
        /*Matrices should be multiples of 8 and MINIMUM 256x256 to make use of AVX256 optimizations
        Otherwise any NxN or NxM sized matrix will work just fine but won't be accelerated through AVX256*/
        contracted_tensor_mul_benchmark(1,15,4096)
    }

    //==========Benchmark output==========:
    Matrix size: 4096x4096
    Tensor batches: 1
    Tensor slices: 15
    2061.58 GFLOP
    AVX CONTRACTED TENSOR MUL: 8.1893s, GFLOP/S = 251.741

- Batched TensorMul Benchmark:
    ```c++
    void batched_tensor_mul_benchmark(size_t batches, size_t slices, size_t matrix_size){
      tens::tensor tensor_a(dims,rank,matrix_size);
      tens::tensor tensor_b(dims,rank,matrix_size);
      tens::tensor_ops tensor_op_a(tensor_a);
      tens::tensor_ops tensor_op_b(tensor_b); 
      tens::tensor_ops::fill_tensor(tensor_op_a);
      tens::tensor_ops::fill_tensor(tensor_op_b);
      std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
      std::cout << "Tensor batches: " << tensor_a.m_batches << std::endl; 
      std::cout << "Tensor slices: " << tensor_a.m_slices << std::endl;
      double totalOps = tensor_a.m_slices * (2 * double(matrix_size) * double(matrix_size) * double(matrix_size));
      double gflopFactor = 1.0e-9;
      std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl;
      auto start = nanos();
      tens::tensor_ops tensor_c = tens::tensor_ops::batch_tensor_mul(tensor_op_a, tensor_op_b);
      auto end = nanos();
      double optTime = (end - start) * 1e-9;
      double optGflops = (totalOps * gflopFactor) / optTime;
      std::cout << "AVX BATCHED TENSOR MUL: " << optTime
                << "s, GFLOP/S = " << optGflops << "\n";
    }
    
    int main(){
        //First argument is the total number of tensor batches
        //The second argument is the total number of tensor slices
        //The third argument is the size of the matrices in each tensor slice
        /*Matrices should be multiples of 8 and MINIMUM 256x256 to make use of AVX256 optimizations
        Otherwise any NxN or NxM sized matrix will work just fine but won't be accelerated through AVX256*/
        batched_tensor_mul_benchmark(1,15,4096)
    }

    //==========Benchmark output==========:
    Matrix size: 4096x4096
    Tensor batches: 1
    Tensor slices: 15
    2061.58 GFLOP
    AVX BATCHED TENSOR MUL: 8.66689s, GFLOP/S = 237.869
    
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
      op1 = mat::mat_ops::transpose_matrix(op1);
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
