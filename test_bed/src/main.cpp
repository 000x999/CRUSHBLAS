#include "../../core/include/matrix_core/matrix.hpp"
#include <stdlib.h>
#include <x86intrin.h>

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}

void MatMulBenchmark(float A){
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
  //std::cout<<"\n=================================="<<std::endl;
  //op1.display(); 
  //std::cout<<"\n=================================="<<std::endl;
  //op2.display();
  //std::cout<<"\n=================================="<<std::endl;
  auto start = nanos(); 
  mat::mat_ops op3 = mat::mat_ops::mat_mul(op1,op2);
  //op3.display(); 
  //std::cout<<"\n=================================="<<std::endl;
  auto end = nanos();
  //op3.display();
  double optTime = (end - start) * 1e-9;
  double optGflops = (totalOps * gflopFactor) / optTime;
  std::cout << "AVX MatMul: " << optTime
            << "s, GFLOP/S = " << optGflops << "\n";
}

void TransposeBenchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
  double totalOps =  double(A) * double(A);
  double memfactor = 2.0 * A *  A * sizeof(float);
  double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
  std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
  mat::matrix mat1(A, A);
  mat::mat_ops op1(mat1); 
  op1.fill_mat();
  auto start = nanos();
  //op1 = op1.transpose();
  auto end = nanos(); 
  double optTime = (end - start) * 1e-9;
  double optmem =  memfactorgb / optTime;
  std::cout << "AVX Transpose: " << optTime
              << "s, GB/S = " << optmem << "\n";
}

void DiagonalBenchmark(float A){
  std::cout<<"Matrix size: " << A << "x"<<A<<std::endl;
  double totalOps =  double(A); 
  double memfactor = 2.0 * A *  A * sizeof(float);
  double memfactorgb = memfactor / (1024.0 * 1024.0 * 1024.0); 
  std::cout<< totalOps * 1e-6<< " KB" << std::endl; 
  mat::matrix test(A,A);
  mat::mat_ops op1(test);
  op1.fill_mat();
  op1.display(); 
  auto start = nanos();
  op1 = op1.return_diagonal();
  auto end = nanos(); 
  op1.display();
  double optTime = (end - start) * 1e-9;
  double optmem =  memfactorgb / optTime;
  std::cout << "\nReturn diagonal: " << optTime
              << "s, GB/S = " << optmem << "\n";
}

int main(){     
  //MatMulBenchmark(4096);
  
  mat::matrix A(5,5); 
  mat::matrix B(5,5); 

  mat::mat_ops op1(A); 
  mat::mat_ops op2(B); 
  op1.fill_mat(); 
  op1.display(); 
  std::cout<<"====================================================="<<std::endl;
  op2.display(); 
  std::cout<<"====================================================="<<std::endl;
  std::cout<<"Checking for AB = I"<<std::endl; 
  mat::mat_ops op3 = mat::mat_ops::mat_mul(op1, op2); 
  op3.display();
  


  /*
  mat::matrix A(5,5); 
  mat::mat_ops op1(A);
  op1.fill_mat(); 
  std::cout<<"=================================="<<std::endl; 
  op1.display(); 
  std::cout<<"=================================="<<std::endl;
  std::cout<< "index[0][0]: "<<op1.return_value(0,0)<< " | "<< " index[5][5]: " << op1.return_value(5,5)<< " | " << " index[3][2]: " 
           << op1.return_value(3,2)<< " | " << " index[1][4]: " << op1.return_value(1,4)<< " | " << std::endl; 
  */

  //TransposeBenchmark(4096);
  //DiagonalBenchmark(2048);
}
