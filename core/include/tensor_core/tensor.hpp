#ifndef TENSOR_H 
#define TENSOR_H 
#include "../logger_core/logger.h"
#include "../matrix_core/matrix.hpp"
#include "../../BLAS/level3/level3.hpp"
#include "../../defines.h" 

namespace tens{
typedef struct tensor{
  size_t m_batches;
  size_t m_slices;
  size_t m_matrix_size; 
  __attribute__((aligned(32))) std::vector<mat::mat_ops> m_data;
  tensor(size_t batches_in, size_t slices_in, size_t matrix_size_in):  m_batches(batches_in), m_slices(slices_in), m_matrix_size(matrix_size_in){
    size_t total_tensor_size = 1; 
    for(size_t i = 0; i < batches_in; ++i){  
      total_tensor_size *= slices_in; 
    }
    m_data.reserve(total_tensor_size);
    mat::matrix dim_matrix_size(matrix_size_in, matrix_size_in);
    for(size_t i = 0; i < total_tensor_size; ++i){
      m_data.emplace_back(mat::mat_ops(dim_matrix_size)); 
    }
  }
}tensor; 

class tensor_ops{
private: 
public: 
  __attribute__((aligned(32))) tens::tensor tensor;
  tensor_ops(tens::tensor &tensor_in): tensor(tensor_in){}
  
  static CRUSH_API tensor_ops reshape_tensor(tensor_ops &tensor_in, size_t reshape_in){
    size_t new_tensor_size = 1; 
    for(size_t i = 0; i < reshape_in; ++i){
      new_tensor_size *= reshape_in; 
    }
  #if DEBUG
    if(new_tensor_size != tensor_in.tensor.m_data.size()){
      CRUSH_FATAL("RESHAPE MUST NOT CHANGE THE TOTAL NUMBER OF ELEMENTS : ASSERTION FAILED"); 
    }
    assert(new_tensor_size == tensor_in.tensor.m_data.size()); 
  #endif
    tensor_in.tensor.m_batches = reshape_in;
    return tensor_in;
  }
  
  #if USE_AVX256
  static CRUSH_API tensor_ops batch_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
  #if DEBUG
    if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){ 
      CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH : ASSERTION FAILED"); 
    }
    assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
  #endif
    size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
    tens::tensor c_tensor(right_tensor.tensor.m_batches, right_tensor.tensor.m_slices, right_tensor.tensor.m_matrix_size);    
    tens::tensor_ops c_tensor_ops(c_tensor); 
    for(size_t i = 0; i < d_tensor_size; ++i){
     c_tensor_ops.tensor.m_data[i].zero_mat();
     c_tensor_ops.tensor.m_data[i] = level3::gemm(0,0,0,left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 0, c_tensor.m_data[i]);
    }
    return c_tensor_ops; 
  }
  #else 
  static CRUSH_API tensor_ops batch_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
  #if DEBUG
    if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){
      CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH : ASSERTION FAILED"); 
    }
    assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
  #endif
    size_t n_tensor_size = right_tensor.tensor.m_data[0].mat.m_row;  
    size_t d_tensor_size = right_tensor.tensor.m_data.size();
    std::cout<<d_tensor_size<<std::endl;
    tens::tensor c_tensor(right_tensor.tensor.m_batches,right_tensor.tensor.m_slices, right_tensor.tensor.m_matrix_size);
    tens::tensor_ops c_tensor_ops(c_tensor);
    for(size_t i = 0; i < d_tensor_size; ++i){
      c_tensor_ops.tensor.m_data[i].zero_mat();
      c_tensor_ops.tensor.m_data[i] = level3::gemm(n_tensor_size, n_tensor_size, n_tensor_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 0, c_tensor.m_data[i]);
    }
    return c_tensor_ops;
  }
  #endif

  #if USE_AVX256
  static CRUSH_API mat::mat_ops contract_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
  #if DEBUG
    if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){
      CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH : ASSERTION FAILED"); 
    }
    assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size());
  #endif
    size_t n_matrix_size = right_tensor.tensor.m_data[0].mat.m_row;
    size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
    mat::matrix c_matrix(n_matrix_size, n_matrix_size);
    mat::mat_ops c_matrix_ops(c_matrix);
    c_matrix_ops.zero_mat();
    for(size_t i = 0; i < d_tensor_size; ++i){
      c_matrix_ops = level3::gemm(0, 0, 0, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 1, c_matrix_ops);
    }
    return c_matrix_ops;
  }
  #else 
  static CRUSH_API mat::mat_ops contract_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
  #if DEBUG
    if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){
      CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH : ASSERTION FAILED"); 
    }
    assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size());
  #endif
    size_t n_matrix_size = right_tensor.tensor.m_data[0].mat.m_row;  
    size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
    mat::matrix c_matrix(n_matrix_size, n_matrix_size);
    mat::mat_ops c_matrix_ops(c_matrix);
    c_matrix_ops.zero_mat();
    for(size_t i = 0; i < d_tensor_size; ++i){
      mat::mat_ops current_buffer = level3::gemm(n_matrix_size, n_matrix_size, n_matrix_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 1, c_matrix_ops);
    }
    return c_matrix_ops;
  }
  #endif 

  static tensor_ops fill_tensor(tensor_ops &tensor_in){
    for(size_t i = 0; i < tensor_in.tensor.m_data.size(); ++i){
      tensor_in.tensor.m_data[i].fill_mat(); 
    }
    return tensor_in; 
  }

  static tensor_ops zero_tensor(tensor_ops &tensor_in){
    for(size_t i = 0; i < tensor_in.tensor.m_data.size(); ++i){
      tensor_in.tensor.m_data[i].zero_mat(); 
    }
    return tensor_in;
  }
};


}; //Namespace tens
    
#endif
