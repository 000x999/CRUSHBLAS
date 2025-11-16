#include "../../../blas/level3/level3.hpp"
#if USE_AVX256
mat::mat_ops level3::blas::gemm(size_t m, size_t n, size_t p, const mat::mat_ops &left_mat, const mat::mat_ops &right_mat, float alpha, float beta, mat::mat_ops &c_mat) {
  constexpr int BLOCK_I = 256;  //1024 bytes at fp32
  constexpr int BLOCK_J = 256;  //1024 bytes at fp32
  constexpr int BLOCK_K = 16;   //64 bytes  at fp32
  mat::matrix A = left_mat.mat;
  mat::matrix B = right_mat.mat;
  mat::matrix C(A.m_row, B.m_col);

  omp_set_num_threads(omp_get_max_threads());
  #pragma omp parallel for collapse(2) schedule(dynamic,1)
  for (size_t i_block = 0; i_block < A.m_row; i_block += BLOCK_I) {
    for (size_t j_block = 0; j_block < B.m_col; j_block += BLOCK_J) {
      float c_buffer[BLOCK_I][BLOCK_J] __attribute__((aligned(32))) = {{0}};
      const size_t i_end = std::min(i_block + BLOCK_I, A.m_row);
      const size_t j_end = std::min(j_block + BLOCK_J, B.m_col);

      if (beta == 0.0f){
        for (size_t ii = 0; ii < i_end - i_block; ++ii)
          for (size_t jj = 0; jj < j_end - j_block; ++jj)
            c_buffer[ii][jj] = 0.0f;
      }else{
        for (size_t ii = i_block; ii < i_end; ++ii) {
          for (size_t jj = j_block; jj < j_end; ++jj) {
            c_buffer[ii - i_block][jj - j_block] = beta * C[ii][jj];
          }
        }
      }
      for (size_t k_block = 0; k_block < A.m_col; k_block += BLOCK_K) {
        const size_t k_end = std::min(k_block + BLOCK_K, A.m_col);
        
        for (size_t i = i_block; i < i_end; ++i) {
          for (size_t k = k_block; k < k_end; ++k) {
            float a_val = A[i][k];
            __m256 alpha_vec = _mm256_set1_ps(alpha);  
            __m256 a_vec = _mm256_broadcast_ss(&a_val);
            if (alpha != 1.0f) {
              a_vec = _mm256_mul_ps(alpha_vec, a_vec);
            }
            size_t j = j_block; 
            //8x unrolled loop
            //128x128 sized matrices MIN No reason to have it any smaller
            /*If smaller matrices than 128x128 are needed, no reason to use with avx, 
              it's fast enough without for smaller sized ones*/
            for (; j + 67 < j_end; j += 64) {
              __m256 b_vec0 = _mm256_load_ps(&B[k][j]);
              __m256 b_vec1 = _mm256_load_ps(&B[k][j +  8]);
              __m256 b_vec2 = _mm256_load_ps(&B[k][j + 16]);
              __m256 b_vec3 = _mm256_load_ps(&B[k][j + 24]);
              __m256 b_vec4 = _mm256_load_ps(&B[k][j + 32]);
              __m256 b_vec5 = _mm256_load_ps(&B[k][j + 40]);
              __m256 b_vec6 = _mm256_load_ps(&B[k][j + 48]);
              __m256 b_vec7 = _mm256_load_ps(&B[k][j + 56]);

              __m256 c_vec0 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block]);
              __m256 c_vec1 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block +  8]);
              __m256 c_vec2 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 16]);
              __m256 c_vec3 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 24]);
              __m256 c_vec4 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 32]);
              __m256 c_vec5 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 40]);
              __m256 c_vec6 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 48]);
              __m256 c_vec7 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 56]);

              c_vec0 = _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);
              c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
              c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
              c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);
              c_vec4 = _mm256_fmadd_ps(a_vec, b_vec4, c_vec4);
              c_vec5 = _mm256_fmadd_ps(a_vec, b_vec5, c_vec5);
              c_vec6 = _mm256_fmadd_ps(a_vec, b_vec6, c_vec6);
              c_vec7 = _mm256_fmadd_ps(a_vec, b_vec7, c_vec7);

              _mm256_store_ps(&c_buffer[i - i_block][j - j_block], c_vec0);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block +  8], c_vec1);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 16], c_vec2);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 24], c_vec3);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 32], c_vec4);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 40], c_vec5);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 48], c_vec6);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 56], c_vec7);
            }
            for(; j + 7 < j_end; j += 8){
              __m256 b_vec = _mm256_loadu_ps(&B[k][j]); 
              float *c_row = &c_buffer[i - i_block][j - j_block]; 
              __m256 c_vec = _mm256_load_ps(c_row);
              c_vec        = _mm256_fmadd_ps(a_vec, b_vec, c_vec); 
              _mm256_store_ps(c_row, c_vec); 
            }
            for(; j < j_end; ++j){
              c_buffer[i - i_block][j - j_block] += a_val * B[k][j] * alpha;
            }
          }
        }
      }
      //flush buffer C
      for (size_t i = i_block; i < i_end; ++i) {
        for (size_t j = j_block; j < j_end; ++j) {
          C[i][j] = c_buffer[i - i_block][j - j_block];
        }
      }
    }
  }
#if DEBUG
  DEBUG_THREADS();
#endif
  return mat::mat_ops(C);
}
#else
mat::mat_ops level3::blas::gemm(size_t m, size_t n, size_t p, const mat::mat_ops &A, const mat::mat_ops &B, float alpha, float beta, mat::mat_ops &C){
  for(size_t i = 0; i < m; ++i){
    for(size_t j = 0; j < n; ++j){
      float result = 0.0f;  
      for(size_t k = 0; k < p; ++k){
        result = result + A.mat[i][k] * B.mat[k][j]; 
        C.mat[i][j] = alpha * result + beta * C.mat[i][j]; 
      }
    }
  }
  return C;  
}
#endif

#if USE_AVX256 
void level3::blas::gemm(size_t m, size_t n, size_t p, const level3::mat_ops_view &left_view, const level3::mat_ops_view &right_view, float alpha, float beta, level3::mat_ops_view &c_view){
  constexpr int BLOCK_I = 256;  //1024 bytes at fp32
  constexpr int BLOCK_J = 256;  //1024 bytes at fp32
  constexpr int BLOCK_K = 16;   //64 bytes  at fp32
  mat_ops_view  A = left_view;
  mat_ops_view  B = right_view;
  mat_ops_view  C = c_view;
//  std::cout << "finished assigning new op views" << '\n'; 
  omp_set_num_threads(omp_get_max_threads());
  #pragma omp parallel for collapse(2) schedule(dynamic,1)
  for (size_t i_block = 0; i_block < A.row_view; i_block += BLOCK_I) {
    for (size_t j_block = 0; j_block < B.row_view; j_block += BLOCK_J) {
      float c_buffer[BLOCK_I][BLOCK_J] __attribute__((aligned(32))) = {{0}};
      const size_t i_end = std::min(i_block + BLOCK_I, A.row_view);
      const size_t j_end = std::min(j_block + BLOCK_J, B.row_view);

      if (beta == 0.0f){
        for (size_t ii = 0; ii < i_end - i_block; ++ii)
          for (size_t jj = 0; jj < j_end - j_block; ++jj)
            c_buffer[ii][jj] = 0.0f;
      }else{
        for (size_t ii = i_block; ii < i_end; ++ii) {
          for (size_t jj = j_block; jj < j_end; ++jj) {
            c_buffer[ii - i_block][jj - j_block] = beta * C[ii][jj];
          }
        }
      }
  //    std::cout << "k block loop start" << '\n';      
      for (size_t k_block = 0; k_block < A.row_view; k_block += BLOCK_K) {
        const size_t k_end = std::min(k_block + BLOCK_K, A.row_view);
        
        for (size_t i = i_block; i < i_end; ++i) {
          for (size_t k = k_block; k < k_end; ++k) {  
    //        std::cout << "a val loading start" << '\n';
            float a_val = A[i][k];
            __m256 alpha_vec = _mm256_set1_ps(alpha);  
            __m256 a_vec     = _mm256_broadcast_ss(&a_val);
            if (alpha != 1.0f) {
              a_vec = _mm256_mul_ps(alpha_vec, a_vec);
            }
            size_t j = j_block; 
            //8x unrolled loop
            //128x128 sized matrices MIN No reason to have it any smaller
            /*If smaller matrices than 128x128 are needed, no reason to use with avx, 
              it's fast enough without for smaller sized ones*/
   //       std::cout << "unrolled loop start" << '\n'; 
            for (; j + 67 < j_end; j += 64) {
              __m256 b_vec0 = _mm256_load_ps(&B[k][j]);
              __m256 b_vec1 = _mm256_load_ps(&B[k][j +  8]);
              __m256 b_vec2 = _mm256_load_ps(&B[k][j + 16]);
              __m256 b_vec3 = _mm256_load_ps(&B[k][j + 24]);
              __m256 b_vec4 = _mm256_load_ps(&B[k][j + 32]);
              __m256 b_vec5 = _mm256_load_ps(&B[k][j + 40]);
              __m256 b_vec6 = _mm256_load_ps(&B[k][j + 48]);
              __m256 b_vec7 = _mm256_load_ps(&B[k][j + 56]);

              __m256 c_vec0 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block]);
              __m256 c_vec1 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block +  8]);
              __m256 c_vec2 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 16]);
              __m256 c_vec3 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 24]);
              __m256 c_vec4 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 32]);
              __m256 c_vec5 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 40]);
              __m256 c_vec6 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 48]);
              __m256 c_vec7 = _mm256_load_ps(&c_buffer[i - i_block][j - j_block + 56]);

              c_vec0 = _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);
              c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
              c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
              c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);
              c_vec4 = _mm256_fmadd_ps(a_vec, b_vec4, c_vec4);
              c_vec5 = _mm256_fmadd_ps(a_vec, b_vec5, c_vec5);
              c_vec6 = _mm256_fmadd_ps(a_vec, b_vec6, c_vec6);
              c_vec7 = _mm256_fmadd_ps(a_vec, b_vec7, c_vec7);

              _mm256_store_ps(&c_buffer[i - i_block][j - j_block], c_vec0);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block +  8], c_vec1);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 16], c_vec2);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 24], c_vec3);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 32], c_vec4);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 40], c_vec5);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 48], c_vec6);
              _mm256_store_ps(&c_buffer[i - i_block][j - j_block + 56], c_vec7);
            }
           // std::cout << "handling excess from unrolled loop" << '\n'; 
            for(; j + 7 < j_end; j += 8){
              __m256 b_vec = _mm256_loadu_ps(&B[k][j]); 
              float *c_row = &c_buffer[i - i_block][j - j_block]; 
              __m256 c_vec = _mm256_load_ps(c_row);
              c_vec        = _mm256_fmadd_ps(a_vec, b_vec, c_vec); 
              _mm256_store_ps(c_row, c_vec); 
            }
            for(; j < j_end; ++j){
              c_buffer[i - i_block][j - j_block] += a_val * B[k][j] * alpha;
            }
          }
        }
      }
      //flush buffer C
      //std::cout << "flushing to c buffer start" << '\n'; 
      for (size_t i = i_block; i < i_end; ++i) {
        for (size_t j = j_block; j < j_end; ++j) {
        //  std::cout << "C[i][j]" << C[i][j] <<  " = " << "c_buffer[i - i_block][j - j_block]" << c_buffer[i - i_block][j - j_block] << '\n'; 
          C[i][j] = c_buffer[i - i_block][j - j_block];
        }
      }
    }
  }
#if DEBUG
  DEBUG_THREADS();
#endif
  //std::cout << "returning final view" << '\n'; 
} 
#endif 
