#include "include/matrix.hpp"
#include <stdlib.h>
#include <x86intrin.h>

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

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
  mat::mat_ops op3 = op1 * op2; 
  auto end = nanos(); 
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX MatMul: " << optTime
              << "s, GFLOP/S = " << optGflops << "\n";
}

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
  //MatMulBenchmark(1024*4);
  TransposeBenchmark(1024*8); 
}
