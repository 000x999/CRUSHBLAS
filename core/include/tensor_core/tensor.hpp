#ifndef TENSOR_H 
#define TENSOR_H 
#include "../logger_core/logger.hpp"
#include "../matrix_core/matrix.hpp"
#include "../../BLAS/level3/level3.hpp"
#include "../../defines.h" 

namespace tens{
typedef struct tensor{
  size_t m_batches;
  size_t m_slices;
  size_t m_matrix_size; 
  __attribute__((aligned(32))) std::vector<mat::mat_ops> m_data;
  tensor(size_t batches_in, size_t slices_in, size_t matrix_size_in);
}tensor; 

class tensor_ops{
private: 
public: 
  __attribute__((aligned(32))) tens::tensor tensor;
  tensor_ops(tens::tensor &tensor_in);
  static CRUSH_API tensor_ops reshape_tensor(tensor_ops &tensor_in, size_t reshape_in);
  static CRUSH_API tensor_ops batch_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor);
  static CRUSH_API mat::mat_ops contract_tensor_mul(const tens::tensor_ops &right_tensor, const tens::tensor_ops &left_tensor);
  static tensor_ops fill_tensor(tensor_ops &tensor_in);
  static tensor_ops zero_tensor(tensor_ops &tensor_in);
};

}; //Namespace tens
    
#endif
