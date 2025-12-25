#include "level3/level3.hpp"
#include <stdlib.h>
#include <x86intrin.h> 
#include <immintrin.h> 
uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}
  
void gemm_benchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
  double totalOps = 2.0 * double(A) * double(A) * double(A);
  double gflopFactor = 1.0e-9;
  std::cout<< totalOps * 1e-9 << " GFLOP" << std::endl; 
  auto start = nanos(); 
  auto end = nanos();
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX GEMM: " << optTime
            << "s, GFLOP/S = " << optGflops << "\n";
}

int main(){
  return 0;
}
