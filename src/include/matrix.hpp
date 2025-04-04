#ifndef MATRIX_H 
#define MATRIX_H 
#include <vector> 
#include <random> 
#include <iostream>
#ifdef DEBUG  
  #include <cassert>
#endif
#ifdef USE_AVX256
  #include <immintrin.h>
  extern "C" int omp_get_thread_num(); 
  extern "C" int omp_get_num_threads();
  extern "C" int omp_set_dynamic(int threads);
  extern "C" int omp_set_num_threads(int threads);
  extern "C" int omp_get_max_threads();
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
#if DEBUG
    float &operator[](size_t col){
      assert(col < m_cols && "COL index OB");  
      return m_start_row[m_cols];
    }
#else 
    float &operator[](size_t col){
      return m_start_row[col];
    }
#endif
  private: 
    float *m_start_row; 
    size_t m_cols; 
  };//end map_rows
public:
  matrix(size_t m_row, size_t m_col) : m_row(m_row), m_col(m_col), m_data(m_row * m_col){}
  map_rows operator[](size_t row) {return map_rows(&m_data[row*m_col], m_col);}
};//end mat  


class mat_ops{

private: 
  int block_i = 32; 
  int block_j = 32; 
  int block_k = 32;
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
        mat[i][j] = 0; 
      }
    }
  }
  
#if USE_AVX256
  mat_ops operator*(const mat_ops &rhs)const{
    std::cout<<"DEBUG: AVX_MATMUL_STARTED"<<std::endl;
    #define N static_cast<int>(this->mat.m_row)
    mat::matrix A = this->mat; 
    mat::matrix B = rhs.mat;
    mat::matrix C(this->mat.m_row, this->mat.m_col);
    mat::matrix C_ref(this->mat.m_row, this->mat.m_col);
    
    #pragma omp parallel for collapse(2) schedule(dynamic) 
    for(int index = 0; index < N; index += block_i){
      for(int jindex = 0; jindex < N; jindex += block_j){
        int i_block_size = (index + block_i <= N) ? block_i : (N - index); 
        int j_block_size = (jindex + block_j <= N) ? block_j : (N - jindex); 

        float c_block[block_i][block_j] __attribute__((aligned(32))); 
        for(int itile = 0; itile < i_block_size; ++itile){
          for(int jtile = 0; jtile < j_block_size; ++jtile){
            c_block[itile][jtile] = 0.0f; 
          }
        }
        for(int kindex = 0; kindex < N; kindex += block_k){
          int k_block_size = (kindex + block_k <= N) ? block_k : (N - kindex);
          for(int itile = 0; itile < i_block_size; ++itile){
            int row_a = index + itile; 
            for(int jtile = 0; jtile < j_block_size; jtile+=8){
              __m256 sum_vector = _mm256_load_ps(&c_block[itile][jtile]); 
              for(int ktile = 0; ktile < k_block_size; ++ktile){
                int k_index = kindex + ktile; 
                __m256 a_val = _mm256_broadcast_ss(&A[row_a][k_index]); 
                __m256 b_val = _mm256_load_ps(&B[k_index][jindex + jtile]); 

                sum_vector= _mm256_fmadd_ps(a_val, b_val, sum_vector);
              }
              _mm256_store_ps(&c_block[itile][jtile], sum_vector); 
            }
          }
        }
        for(int itile = 0; itile < i_block_size; ++itile){
          int row_c = index + itile; 
          for(int jtile = 0; jtile < j_block_size; ++jtile){
            C[row_c][jindex + jtile] = c_block[itile][jtile]; 
          }
        }
      } 
    }
    mat_ops ops(C);
    return ops;
  }
#else 
  mat_ops operator*(const mat_ops &rhs) const{
    mat_ops temp_mat = rhs;
    mat::matrix A = rhs.mat; 
    std::cout<<"DEBUG: std_matmul_started"<<std::endl;
    for(int i = 0; i < rhs.mat.m_row; ++i){
      for(int j = 0; j < rhs.mat.m_col; ++j){
        for(int k = 0; k < rhs.mat.m_col; ++k){
          temp_mat.mat[i][j] = A[i][k] * A[k][j]; 
        }
      }
    }
    return temp_mat;
  } 
#endif

#if USE_AVX256
  mat_ops transpose(){
    mat::matrix res(this->mat.m_col, this->mat.m_row); 
    float block[8][8] __attribute__((aligned(32))); 
    for(size_t i = 0; i < this->mat.m_row; i += 8){
      for(size_t j = 0; j < this->mat.m_col; j += 8){
        for(size_t iblock = 0; iblock < 8; ++iblock){
          _mm256_store_ps(block[iblock], _mm256_load_ps(&(*this).mat[i+iblock][j]));
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
  mat_ops transpose(){
    mat::matrix temp_mat(this->mat.m_row, this->mat.m_col); 
    for(size_t i = 0; i < this->mat.m_row; i++){
      for(size_t j = 0; j < this->mat.m_col; j++){
        temp_mat[i][j] = this->mat[i][j]; 
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
#endif

};//end mat_ops 

};//End namespace 
#endif 

