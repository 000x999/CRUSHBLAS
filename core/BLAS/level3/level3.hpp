#ifndef LEVEL3_H
#define LEVEL3_H 
#include <cstring>
#include <thread>
#include <immintrin.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <new> 
#include <algorithm> 
#include<cassert> 

#ifdef _WIN32 
#include <malloc.h> 
#else 
#include <stdlib.h>
#endif

#include <queue>
#include <functional>
#include <vector>
#include <memory>
#include <immintrin.h>
#include "crush_defines.h"

namespace level3{
enum class transpose_gemm{
  no_transpose, 
  transpose
};

inline bool eno_trans(transpose_gemm flag){ return flag == transpose_gemm::no_transpose; }

struct mat_ops_view{
  size_t row_view; 
  size_t col_view; 
  size_t leading_dimension;
  float  *data_view;

  class map_view{ 
  public:
    map_view                          (float *m_start_row, size_t m_cols);
    map_view                          (const float *m_start_row, size_t m_cols);

    float       &operator[]           (size_t col); 
    const float &operator[]           (size_t col) const;
 
  private: 
    float *m_start_row; 
    size_t m_cols; 
  };
  
  inline       float    &operator()   (size_t i, size_t j);
  inline const float    &operator()   (size_t i, size_t j) const;
 
  inline       map_view operator[]    (size_t row);
  inline const map_view operator[]    (size_t row)const; 
}; 



class gemm_thread_pool{
public:
  static gemm_thread_pool &get_instance  (); 
  void                     enqueue_task  ( std::function<void()> task_in );
  void                     wait_for_all  (); 
  void                     init_pool     ( size_t num_threads ); 
  void                     shutdown_pool ();

private:
   gemm_thread_pool(); 
  ~gemm_thread_pool();
  std::vector<std::thread>          pool_workers; 
  std::queue<std::function<void()>> worker_tasks; 
  std::mutex                        queue_mutex; 
  std::condition_variable           condition; 
  std::condition_variable           finished; 
  std::atomic<size_t>               active_tasks; 
  bool                              stop; 

  void                              worker_thread(); 
};


class blas{
public:

  //static mat::mat_ops gemm                        (size_t m, size_t n, size_t p, const mat::mat_ops &left_mat, const mat::mat_ops &right_mat, float alpha, float beta, mat::mat_ops &c_mat);
  static void                 gemm                        (size_t m, size_t n, size_t p, const mat_ops_view &left_view, const mat_ops_view &right_view, float alpha, float beta, mat_ops_view &c_view);
  static inline void          pack_left_block             (transpose_gemm transpose_left, const mat_ops_view &left_view, size_t index_zero, size_t kindex_zero, size_t m_c, size_t k_c, float *left_pack); 
  static inline void          pack_right_block            (transpose_gemm transpose_right, const mat_ops_view &right_view, size_t kindex_zero, size_t jindex_zero, size_t k_c, size_t n_c, float *right_pack);

  static inline void          microkernel_4x8_avx256      (const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta);
  static inline void          microkernel_eno_edge        (size_t m_r, size_t n_r, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta); 
  static inline void          microkernel_6x16_avx256     (const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta);

  static inline void          transpose_8x8_ps            (__m256 &row_zero, __m256 &row_one, __m256 &row_two, __m256 &row_three, __m256 &row_four, __m256 &row_five, __m256 &row_six, __m256 &row_seven);
  static void                 gemm_avx256                 (transpose_gemm transpose_left, transpose_gemm transpose_right, size_t m, size_t n, size_t p, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, float alpha = 1.0f, float beta = 0.0f);  
  static void                 gemm_avx256_mt              (transpose_gemm transpose_left, transpose_gemm transpose_right, size_t m, size_t n, size_t p, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, float alpha = 1.0f, float beta = 0.0f);
  static void                 gemm_avx256_core            (transpose_gemm transpose_left, transpose_gemm transpose_right, size_t m, size_t n, size_t p, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c,  size_t i_begin, size_t i_end, float alpha = 1.0f, float beta = 0.0f);
  
  static inline size_t        pad_to_multiple             (size_t value, size_t multiple) { return ((value + multiple - 1) / multiple) * multiple; }

  static void                 crush_gemm                  (transpose_gemm transpose_left, transpose_gemm transpose_right, const mat_ops_view &left_view, const mat_ops_view &right_view, float alpha, float beta, mat_ops_view &c_view);

  static level3::mat_ops_view softmax                     (level3::mat_ops_view &input_view); 
}; //blas 
}; //Level3 namespace

#endif 
