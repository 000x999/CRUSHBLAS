#ifndef MATRIX_H 
#define MATRIX_H 
#include "../logger_core/logger.hpp"
#include "../../defines.h"

#define MATRIX
namespace mat{
struct matrix{
public:
  size_t m_row; 
  size_t m_col;
  __attribute__((aligned(32))) std::vector<float> m_data;
  class map_rows{
  public: 
    map_rows(float* m_start_row, size_t m_cols);
    map_rows(const float *m_start_row, size_t m_cols);
    
    float &operator[](size_t col);
    const float &operator[](size_t col) const; 
  private: 
    float *m_start_row; 
    size_t m_cols; 
  };//end map_rows
public:
  matrix() = default; 
  matrix(size_t m_row, size_t m_col) : m_row(m_row), m_col(m_col), m_data(m_row * m_col){}
  map_rows operator[](size_t row);
  const map_rows operator[](size_t row) const;
};//end mat  

class mat_ops{
private: 
public: 
  __attribute__((aligned(32))) mat::matrix mat;
  //size_t matrix_size; 
  mat_ops() = default;
  //mat_ops(size_t matrix_size) : matrix_size(matrix_size){mat = mat::matrix(matrix_size,matrix_size);} 
  mat_ops(mat::matrix &mat); 
  void display();
  void fill_mat();
  void zero_mat();
  uint64_t nanos(); 
  size_t return_row_count()const;
  size_t return_col_count()const;
  static CRUSH_API mat_ops mat_mul(const mat_ops &left_mat, const mat_ops &right_mat);
  static CRUSH_API mat_ops transpose_matrix(const mat_ops &mat_in);
  static CRUSH_API mat_ops block_matrix(const mat_ops &mat_in, size_t i, size_t j, size_t p, size_t q);
  static CRUSH_API mat_ops add_matrix(const mat_ops &right_mat, const mat_ops &left_mat); 
};//end mat_ops 

};//End namespace 
#endif 

