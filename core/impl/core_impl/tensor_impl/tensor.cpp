#include "../../../include/tensor_core/tensor.hpp"

tens::tensor::tensor(size_t batches_in, size_t slices_in, size_t matrix_size_in):m_batches(batches_in), m_slices(slices_in), m_matrix_size(matrix_size_in){
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

tens::tensor_ops::tensor_ops(tens::tensor &tensor_in): tensor(tensor_in){}
CRUSH_API tens::tensor_ops tens::tensor_ops::reshape_tensor(tensor_ops &tensor_in, size_t reshape_in){
  size_t new_tensor_size = 1; 
  for(size_t i = 0; i < reshape_in; ++i){
    new_tensor_size *= reshape_in; 
  }
#if DEBUG
  if(new_tensor_size != tensor_in.tensor.m_data.size()){
    CRUSH_FATAL("RESHAPE MUST NOT CHANGE THE TOTAL NUMBER OF ELEMENTS :: ASSERTION FAILED"); 
  }
  assert(new_tensor_size == tensor_in.tensor.m_data.size()); 
#endif
  tensor_in.tensor.m_batches = reshape_in; 
  return tensor_in; 
}

#if USE_AVX256 
CRUSH_API tens::tensor_ops tens::tensor_ops::batch_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
#if DEBUG
  if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){ 
    CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH :: ASSERTION FAILED"); 
  }
  assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
#endif
  size_t n_tensor_size = right_tensor.tensor.m_data[0].mat.m_row; 
  size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
  tens::tensor c_tensor(right_tensor.tensor.m_batches, right_tensor.tensor.m_slices, right_tensor.tensor.m_matrix_size); 
  tens::tensor_ops c_tensor_ops(c_tensor); 
  for(size_t i = 0; i < d_tensor_size; ++i){
    c_tensor_ops.tensor.m_data[i].zero_mat();
    c_tensor_ops.tensor.m_data[i] = level3::blas::gemm(n_tensor_size, n_tensor_size, n_tensor_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 0, c_tensor.m_data[i]);
  }
  return c_tensor_ops;
}
#else
CRUSH_API tens::tensor_ops tens::tensor_ops::batch_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
#if DEBUG
  if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){ 
    CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH :: ASSERTION FAILED"); 
  }
  assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
#endif
  size_t n_tensor_size = right_tensor.tensor.m_data[0].mat.m_row; 
  size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
  tens::tensor c_tensor(right_tensor.tensor.m_batches, right_tensor.tensor.m_slices, right_tensor.tensor.m_matrix_size); 
  tens::tensor_ops c_tensor_ops(c_tensor); 
  for(size_t i = 0; i < d_tensor_size; ++i){
    c_tensor_ops.tensor.m_data[i].zero_mat();
    c_tensor_ops.tensor.m_data[i] = level3::blas::gemm(n_tensor_size, n_tensor_size, n_tensor_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 0, c_tensor.m_data[i]);
  }
  return c_tensor_ops;
}
#endif

#if USE_AVX256 
CRUSH_API mat::mat_ops tens::tensor_ops::contract_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
#if DEBUG
  if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){ 
    CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH :: ASSERTION FAILED"); 
  }
  assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
#endif
  size_t n_matrix_size = right_tensor.tensor.m_data[0].mat.m_row; 
  size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
  mat::matrix c_matrix(n_matrix_size, n_matrix_size);
  mat::mat_ops c_matrix_ops(c_matrix); 
  for(size_t i = 0; i < d_tensor_size; ++i){
    c_matrix_ops = level3::blas::gemm(n_matrix_size, n_matrix_size, n_matrix_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 1, c_matrix_ops);
  }
  return c_matrix_ops;
}
#else
CRUSH_API mat::mat_ops tens::tensor_ops::contract_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor){
#if DEBUG
  if(right_tensor.tensor.m_data.size() != left_tensor.tensor.m_data.size()){ 
    CRUSH_FATAL("TENSOR RANK AND SIZES DO NOT MATCH :: ASSERTION FAILED"); 
  }
  assert(right_tensor.tensor.m_data.size() == left_tensor.tensor.m_data.size()); 
#endif
  size_t n_matrix_size = right_tensor.tensor.m_data[0].mat.m_row; 
  size_t d_tensor_size = right_tensor.tensor.m_data.size(); 
  mat::matrix c_matrix(n_matrix_size, n_matrix_size);
  mat::mat_ops c_matrix_ops(c_matrix); 
  for(size_t i = 0; i < d_tensor_size; ++i){
    c_matrix_ops = level3::blas::gemm(n_matrix_size, n_matrix_size, n_matrix_size, left_tensor.tensor.m_data[i], right_tensor.tensor.m_data[i], 1, 1, c_matrix_ops);
  }
  return c_matrix_ops;
}
#endif

#if USE_AVX256
tens::tensor_ops tens::tensor_ops::fill_tensor(tensor_ops &tensor_in){
  size_t tensor_in_size = tensor_in.tensor.m_data.size(); 
  for(size_t i = 0; i < tensor_in_size; ++i){
    tensor_in.tensor.m_data[i].fill_mat(); 
  }
  return tensor_in; 
}
#else
tens::tensor_ops tens::tensor_ops::fill_tensor(tensor_ops &tensor_in){
  size_t tensor_in_size = tensor_in.tensor.m_data.size(); 
  for(size_t i = 0; i < tensor_in_size; ++i){
    tensor_in.tensor.m_data[i].fill_mat(); 
  }
  return tensor_in; 
}
#endif

#if USEAVX256
tens::tensor_ops tens::tensor_ops::zero_tensor(tensor_ops &tensor_in){
  size_t tensor_in_size = tensor_in.tensor.m_data.size(); 
  for(size_t i = 0; i < tensor_in_size; ++i){
    tensor_in.tensor.m_data[i].zero_mat(); 
  }
  return tensor_in; 
}
#else
tens::tensor_ops tens::tensor_ops::zero_tensor(tensor_ops &tensor_in){
  size_t tensor_in_size = tensor_in.tensor.m_data.size(); 
  for(size_t i = 0; i < tensor_in_size; ++i){
    tensor_in.tensor.m_data[i].zero_mat(); 
  }
  return tensor_in; 
}
#endif 
