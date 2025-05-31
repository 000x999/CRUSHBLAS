#include "../../core/include/matrix_core/matrix.hpp"
#include "../../core/BLAS/level3/level3.hpp"
#include "../../core/include/tensor_core/tensor.hpp"
#include <stdlib.h>
#include <x86intrin.h> 
#include <immintrin.h> 
uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void matmul_benchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
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
  mat::mat_ops op3 = mat::mat_ops::mat_mul(op1,op2);
  auto end = nanos();
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX MatMul: " << optTime
            << "s, GFLOP/S = " << optGflops << "\n";
}

void transpose_benchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
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
  std::cout << "AVX Transpose: " << optTime
              << "s, GB/S = " << optmem << "\n";
}

void gemm_benchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
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
  mat::mat_ops op3 = level3::gemm(0,0,0, op1,op2, 1.0f, 0.0f, op3);
  auto end = nanos();
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX GEMM: " << optTime
            << "s, GFLOP/S = " << optGflops << "\n";
}

void contracted_tensor_mul_benchmark(size_t dims, size_t rank, size_t matrix_size){
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

void batched_tensor_mul_benchmark(size_t dims, size_t rank, size_t matrix_size){
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
  std::cout<<"======================================"<<std::endl;
  contracted_tensor_mul_benchmark(1,15,4096);
  std::cout<<"======================================"<<std::endl;
  batched_tensor_mul_benchmark(1,15,4096); 
  std::cout<<"======================================"<<std::endl;
  return 0;
}
