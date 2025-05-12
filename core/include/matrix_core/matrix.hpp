#ifndef MATRIX_H 
#define MATRIX_H 
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#ifdef USE_AVX256
  #include <immintrin.h>
  extern "C" int omp_get_thread_num(); 
  extern "C" int omp_get_num_threads();
  extern "C" int omp_set_dynamic(int threads);
  extern "C" int omp_set_num_threads(int threads);
  extern "C" int omp_get_max_threads();
#endif 
#ifdef DEBUG  
  #include <cassert>
#define DEBUG_THREADS() do {                                \
     _Pragma("omp parallel")                                \
      printf("Thread %d out of %d (File: %s, Line: %d)\n",  \
             omp_get_thread_num(),                          \
             omp_get_num_threads(),                         \
             __FILE__, __LINE__);                           \
  }while(0)
#endif

#define MATRIX
namespace mat{
struct matrix{
public:
  size_t m_row; 
  size_t m_col;
  __attribute__((aligned(32))) std::vector<float> m_data;
  class map_rows{
  public: 
    map_rows(float* m_start_row, size_t m_cols): m_start_row(m_start_row), m_cols(m_cols){}
    map_rows(const float *m_start_row, size_t m_cols): m_start_row(const_cast<float*>(m_start_row)), m_cols(m_cols){}
    
    float &operator[](size_t col){
    #if DEBUG 
      assert(col < m_cols && "COL index OB");
    #endif
      /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
      ~70GLFOP/s...*/ 
      
      //size_t new_bound = col < m_cols ? col : m_cols - 1; 
      //return m_start_row[new_bound];
      return m_start_row[col]; 
    }
    const float &operator[](size_t col) const{
    #if DEBUG  
      assert(col < m_cols && "COL index OB");
    #endif
      /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
      ~70GLFOP/s...*/       

      //size_t new_bound = col < m_cols ? col : m_cols - 1;      
      //return m_start_row[new_bound];
      return m_start_row[col];    
    }
  private: 
    float *m_start_row; 
    size_t m_cols; 
  };//end map_rows
public:
  matrix(size_t m_row, size_t m_col) : m_row(m_row), m_col(m_col), m_data(m_row * m_col){}
  map_rows operator[](size_t row) {
    /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
      ~70GLFOP/s...*/ 

    //size_t new_bound = row < m_row ? row : m_row - 1;    
    //return map_rows(&m_data[new_bound * m_col], m_col);
    return map_rows(&m_data[row * m_col], m_col); 
  }
  const map_rows operator[](size_t row) const{
    /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
      ~70GLFOP/s...*/ 

    //size_t new_bound = row < m_row ? row : m_row - 1; 
    //return map_rows(&m_data[new_bound * m_col], m_col);
     return map_rows(&m_data[row * m_col], m_col); 
  }
};//end mat  

class mat_ops{
private: 
  __attribute__((aligned(32))) mat::matrix mat;

public: 
  mat_ops(mat::matrix &mat): mat(mat){}
  
  void display(){
    std::cout<<"["<<std::endl;
    for(size_t i = 0; i < mat.m_row; ++i){
      for(size_t j = 0; j < mat.m_col; ++j){
        std::cout<<mat[i][j] <<", "; 
      }
      std::cout<<""<<std::endl;
    }
    std::cout<<"]"; 
  }

  void fill_mat(){
    static std::random_device rand; 
    static std::mt19937 gen(rand());
    static std::uniform_real_distribution<float> random_value(1,9);
    for(size_t i = 0; i < mat.m_row; ++i){
      for(size_t j = 0; j < mat.m_col; ++j){
        mat[i][j] = random_value(gen); 
      }
    }
  } 

  void zero_mat(){
    for(size_t i = 0; i < mat.m_row; ++i){
      for(size_t j = 0; j < mat.m_col; ++j){
        mat[i][j] = 1; 
      }
    }
  }

  float return_value(size_t i, size_t j){
    return this->mat[i][j]; 
  }
  
#if USE_AVX256
 static mat_ops mat_mul(const mat_ops &left_mat, const mat_ops &right_mat){
  #if DEBUG
    assert(left_mat.mat.m_col == left_mat.mat.m_row && right_mat.mat.m_col == right_mat.mat.m_row && "Matrix is not square");
    assert((left_mat.mat.m_col * left_mat.mat.m_row % 128 != 0) && (right_mat.mat.m_col * right_mat.mat.m_row % 128 != 0 ) && "Matrix is not a multiple of 128"); 
  #endif
  constexpr int BLOCK_I = 256; //1024 bytes at fp32
  constexpr int BLOCK_J = 256; //1024 bytes at fp32 
  constexpr int BLOCK_K = 16;  //64 bytes at fp32

  mat::matrix A = left_mat.mat;
  mat::matrix B = right_mat.mat;
  mat::matrix C(A.m_row, B.m_col);

  omp_set_num_threads(omp_get_max_threads());
  #pragma omp parallel for collapse(2) schedule(dynamic,1)
  for(size_t i_block = 0; i_block < A.m_row; i_block += BLOCK_I) {
    for(size_t j_block = 0; j_block < B.m_col; j_block += BLOCK_J) {
      float c_buffer[BLOCK_I][BLOCK_J] __attribute__((aligned(32))) = {{0}};
      
      const size_t i_end = std::min(i_block + BLOCK_I, A.m_row);
      const size_t j_end = std::min(j_block + BLOCK_J, B.m_col);
      
      for(size_t k_block = 0; k_block < A.m_col; k_block += BLOCK_K) {
        const size_t k_end = std::min(k_block + BLOCK_K, A.m_col);
          
        for(size_t i = i_block; i < i_end; ++i) {
          for (size_t k = k_block; k < k_end; ++k) {
            const float a_val = A[i][k];
            __m256 a_vec = _mm256_broadcast_ss(&a_val);
            //8x unrolled loop
            //128x128 sized matrices MIN -- No reason to have it any smaller
            /*If smaller matrices than 128x128 are needed, no reason to use with avx, 
              it's fast enough without for smaller sized ones*/
            for(size_t j = j_block; j < j_end; j += 64) {
              __m256 b_vec0 = _mm256_load_ps(&B[k][j]);
              __m256 b_vec1 = _mm256_load_ps(&B[k][j+8]);
              __m256 b_vec2 = _mm256_load_ps(&B[k][j+16]);
              __m256 b_vec3 = _mm256_load_ps(&B[k][j+24]); 
              __m256 b_vec4 = _mm256_load_ps(&B[k][j+32]);
              __m256 b_vec5 = _mm256_load_ps(&B[k][j+40]);
              __m256 b_vec6 = _mm256_load_ps(&B[k][j+48]);
              __m256 b_vec7 = _mm256_load_ps(&B[k][j+56]); 

              __m256 c_vec0 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block]);
              __m256 c_vec1 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+8]);
              __m256 c_vec2 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+16]);
              __m256 c_vec3 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+24]);
              __m256 c_vec4 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+32]);
              __m256 c_vec5 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+40]);
              __m256 c_vec6 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+48]);
              __m256 c_vec7 = _mm256_load_ps(&c_buffer[i-i_block][j-j_block+56]);
             
              c_vec0 = _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);
              c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
              c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2); 
              c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);
              c_vec4 = _mm256_fmadd_ps(a_vec, b_vec4, c_vec4);
              c_vec5 = _mm256_fmadd_ps(a_vec, b_vec5, c_vec5);
              c_vec6 = _mm256_fmadd_ps(a_vec, b_vec6, c_vec6);
              c_vec7 = _mm256_fmadd_ps(a_vec, b_vec7, c_vec7);
      
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block],     c_vec0);
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+8],   c_vec1);
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+16],  c_vec2);
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+24],  c_vec3);  
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+32],  c_vec4);            
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+40],  c_vec5);
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+48],  c_vec6);
              _mm256_store_ps(&c_buffer[i-i_block][j-j_block+56],  c_vec7);
            }
          }
        }
      }
    //Flush buffer to C
    for(size_t i = i_block; i < i_end; ++i) {
      for(size_t j = j_block; j < j_end; ++j) {
        C[i][j] = c_buffer[i - i_block][j - j_block];
        }
      }
    }
  }
#if DEBUG 
  DEBUG_THREADS();
#endif
  return mat_ops(C);
}
#else 
 static mat_ops mat_mul(const mat_ops &left_mat, const mat_ops &right_mat){
    size_t mat_size_row = left_mat.mat.m_row;
    size_t mat_size_col = left_mat.mat.m_col; 
    size_t mat_col = right_mat.mat.m_col; 
    mat::matrix temp_mat(mat_size_row, mat_col);
  #if DEBUG
    std::cout<<"DEBUG: std_matmul_started"<<std::endl;
  #endif
    for(int i = 0; i < mat_size_row; ++i){
      for(int j = 0; j < mat_col; ++j){
        float sum = 0.0f;
        for(int k = 0; k < mat_size_col; ++k){
          sum += left_mat.mat[i][k] * right_mat.mat[k][j]; 
        }
        temp_mat[i][j] = sum;
      }
    }
    return mat_ops(temp_mat);
  } 
#endif

#if USE_AVX256
  mat_ops transpose(){
    mat::matrix res(this->mat.m_col, this->mat.m_row); 
    float block[8][8] __attribute__((aligned(32))); 
    for(size_t i = 0; i < this->mat.m_row; i += 8){
      for(size_t j = 0; j < this->mat.m_col; j += 8){
        for(size_t iblock = 0; iblock < 8; ++iblock){
          _mm256_store_ps(block[iblock], _mm256_load_ps(&this->mat[i+iblock][j]));
        }
        for(size_t iblock = 0; iblock < 8; ++iblock){
          for(size_t jblock = iblock + 1; jblock < 8; ++jblock){
            std::swap(block[iblock][jblock], block[jblock][iblock]); 
          }
        }
        for(size_t jblock = 0; jblock < 8; ++jblock){
          _mm256_store_ps(&res[j+jblock][i], _mm256_load_ps(block[jblock])); 
        }
      }
    }
    return mat_ops(res); 
  }
#else 
  /*
  static mat_ops transpose(const mat_ops &mat_in){
    mat::matrix temp_mat(mat_in.mat.m_row, mat_in.mat.m_col); 
    for(size_t i = 0; i < mat_in.mat.m_row; i++){
      for(size_t j = 0; j < mat_in.mat.m_col; j++){
        temp_mat[i][j] = mat_in.mat[i][j]; 
      }
    }
    size_t temp = this->mat.m_row; 
    this->mat.m_row = this->mat.m_col; 
    this->mat.m_col = temp;
    for(size_t i = 0; i < this->mat.m_row; i++){
      for(size_t j = 0; j < this->mat.m_col; j++){
        this->mat[i][j] = temp_mat[j][i]; 
      }
    }
    return mat_ops(temp_mat);
  }
        */
#endif

  static mat_ops opp_sign(mat_ops &mat_in){
    for(size_t i = 0; i < mat_in.mat.m_row; ++i){
      for(size_t j = 0; j < mat_in.mat.m_col; ++j){
        mat_in.mat[i][j] = -mat_in.mat[i][j]; 
      }
    }
    return mat_in; 
  }

  static mat_ops subtract_matrix(const mat_ops &mat_left, const mat_ops &mat_right){
    mat::matrix temp_mat(mat_left.mat.m_row, mat_left.mat.m_col); 
    for(size_t i = 0; i < mat_left.mat.m_row; ++i){
      for(size_t j = 0; j < mat_left.mat.m_col; ++j){
        temp_mat[i][j] = mat_left.mat[i][j] - mat_right.mat[i][j];
      }
    }
    return mat_ops(temp_mat);
  } 

  mat_ops return_diagonal(){
    mat::matrix temp_mat(this->mat.m_row, this->mat.m_col);
    #if DEBUG
      assert(this->mat.m_col == this->mat.m_row && "Matrix is not square");
      assert((this->mat.m_col * this->mat.m_row % 128 != 0) && "Matrix is not a multiple of 128"); 
    #endif
    for(size_t i = 0; i < this->mat.m_row; ++i){
      #if DEBUG
        std::cout<< "row_index: "<< i << " col_index: " << j << std::endl; 
      #endif
      temp_mat[i][i] = this->mat[i][i]; 
    }
    return mat_ops(temp_mat); 
  }
 
};//end mat_ops 

};//End namespace 
#endif 

