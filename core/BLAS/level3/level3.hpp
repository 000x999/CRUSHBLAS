#ifndef LEVEL3_H
#define LEVEL3_H 
#include "../../defines.h"
#include "../../include/matrix_core/matrix.hpp"
namespace level3{
struct mat_ops_view{
  size_t row_view; 
  size_t col_view; 
  float  *data_view;

  class map_view{ 
  public:
    map_view(float *m_start_row, size_t m_cols): m_start_row(m_start_row), m_cols(m_cols){}  
    map_view(const float *m_start_row, size_t m_cols): m_start_row(const_cast<float*>(m_start_row)), m_cols(m_cols){}

    float &operator[](size_t col){
    #if DEBUG
      if(!(col < m_cols)){
        CRUSH_FATAL("COL INDEX OUT OF BOUNDS :: ASSERTION FAILED"); 
      }
      assert(col < m_cols); 
    #endif 
      return m_start_row[col]; 
    }

    const float &operator[](size_t col)const{
    #if DEBUG
      if(!(col < m_cols)){
        CRUSH_FATAL("COL INDEX OOB :: ASSERTION FAILED"); 
      }
      assert(col < m_cols);
    #endif   
      return m_start_row[col];     
    }
 
  private: 
    float *m_start_row; 
    size_t m_cols; 
  };

  map_view operator[](size_t row){
  #if DEBUG
    if(!(row_view, row)){
      CRUSH_FATAL("ROW INDEX OOB :: ASSERTION FAILED");
    }
    assert(row_view < row); 
  #endif 
    return map_view(&data_view[row * col_view], col_view); 
  }
  const map_view operator[](size_t row)const{
  #if DEBUG
    if(!(row_view, row)){
      CRUSH_FATAL("ROW INDEX OOB :: ASSERTION FAILED");
    }
    assert(row_view < row); 
  #endif
    return map_view(&data_view[row * col_view], col_view); 
  }

}; 


class blas{
public:
  static mat::mat_ops gemm(size_t m, size_t n, size_t p, const mat::mat_ops &left_mat, const mat::mat_ops &right_mat, float alpha, float beta, mat::mat_ops &c_mat);
  static void gemm(size_t m, size_t n, size_t p, const mat_ops_view &left_view, const mat_ops_view &right_view, float alpha, float beta, mat_ops_view &c_view); 
}; //blas 
}; //Level3 namespace

#endif 
