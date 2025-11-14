#ifndef LEVEL3_H
#define LEVEL3_H 
#include "../../defines.h"
#include "../../include/matrix_core/matrix.hpp"
namespace level3{
class blas{
public:
  static CRUSH_API mat::mat_ops gemm(size_t m, size_t n, size_t p, const mat::mat_ops &left_mat, const mat::mat_ops &right_mat, float alpha, float beta, mat::mat_ops &c_mat);
}; //blas 
}; //Level3 namespace

#endif 
