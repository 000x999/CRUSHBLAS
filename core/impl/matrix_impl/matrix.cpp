#include "matrix_core/matrix.hpp"

mat::matrix::map_rows::map_rows(float *m_start_row, size_t m_cols): m_start_row(m_start_row), m_cols(m_cols){}
mat::matrix::map_rows::map_rows(const float *m_start_row, size_t m_cols): m_start_row(const_cast<float*>(m_start_row)), m_cols(m_cols){}

float &mat::matrix::map_rows::operator[](size_t col){
#if DEBUG
  if(!(col < m_cols)){
    CRUSH_FATAL("COL INDEX OUT OF BOUNDS :: ASSERTION FAILED"); 
  }
  assert(col < m_cols); 
#endif 
  /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
      ~70GLFOP/s...*/ 
  //size_t new_bound = col < m_cols ? col : m_cols - 1; 
  //return m_start_row[new_bound];
  return m_start_row[col]; 
}

const float &mat::matrix::map_rows::operator[](size_t col)const{
#if DEBUG
  if(!(col < m_cols)){
    CRUSH_FATAL("COL INDEX OOB :: ASSERTION FAILED"); 
  }
  assert(col < m_cols);
#endif   
  /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
  ~70GLFOP/s...*/       

  //size_t new_bound = col < m_cols ? col : m_cols - 1;      
  //return m_start_row[new_bound];
  return m_start_row[col];     
}

mat::matrix::map_rows mat::matrix::operator[](size_t row){
  /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
    ~70GLFOP/s...*/ 

  //size_t new_bound = row < m_row ? row : m_row - 1;    
  //return map_rows(&m_data[new_bound * m_col], m_col);
  return map_rows(&m_data[row * m_col], m_col); 
}
const mat::matrix::map_rows mat::matrix::operator[](size_t row)const{
 /*Could add this back in for more 'correct' manual indexing but it slows down the matmul by 
    ~70GLFOP/s...*/ 

  //size_t new_bound = row < m_row ? row : m_row - 1; 
  //return map_rows(&m_data[new_bound * m_col], m_col);
  return map_rows(&m_data[row * m_col], m_col); 
}

mat::mat_ops::mat_ops(mat::matrix &mat): mat(mat) {}

void mat::mat_ops::display(){
  std::cout << "[" << '\n';
  for(size_t i = 0; i < mat.m_row; ++i){
    for(size_t j = 0; j < mat.m_col; ++j){
      std::cout << mat[i][j] << ", "; 
    }
    std::cout << "" << std::endl;
  }
  std::cout << "]"; 
}

uint64_t mat::mat_ops::nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec * 1000000000ULL + (uint64_t)start.tv_nsec;
}


#if USE_AVX256
void mat::mat_ops::fill_mat(){
#if DEBUG
  auto start = nanos(); 
#endif
static std::random_device rand; 
  static std::mt19937 gen(rand());
  static std::uniform_real_distribution<float> random_value(1.0f, 9.0f);
  for(size_t i = 0; i < mat.m_row; ++i){
    size_t j = 0;

    for(; j + 7 < mat.m_col; j += 8){
      alignas(32) float random[8];

      for(int k = 0; k < 8; ++k){
        random[k] = random_value(gen);
      }
      __m256 random_vector = _mm256_load_ps(random);
      _mm256_storeu_ps(&mat[i][j], random_vector); 
    }

    for(; j < mat.m_col; ++j){
      mat[i][j] = random_value(gen); 
    }
 }
#if DEBUG
  auto end = nanos(); 
  std::cout << "Matrix fill time: " << (end - start) * 1e-9 << '\n'; 
#endif
}
#else
void mat::mat_ops::fill_mat(){
#if DEBUG
  auto start = nanos(); 
#endif 
  static std::random_device rand; 
  static std::mt19937 gen(rand());
  static std::uniform_real_distribution<float> random_value(1,9);
  for(size_t i = 0; i < mat.m_row; ++i){
    for(size_t j = 0; j < mat.m_col; ++j){
      mat[i][j] = random_value(gen); 
    }
  }
#if DEBUG
  auto end = nanos(); 
  std::cout << "Matrix fill time: " << (end - start) * 1e-9 << '\n'; 
#endif 
}
#endif 

#if USE_AVX256
void mat::mat_ops::zero_mat(){
  for(size_t i = 0; i < mat.m_row; ++i){
    size_t j = 0;

    for(; j + 7 < mat.m_col; j += 8){
      alignas(32) float random[8];

      for(int k = 0; k < 8; ++k){
        random[k] = 0.0f; 
      }
      __m256 random_vector = _mm256_load_ps(random);
      _mm256_storeu_ps(&mat[i][j], random_vector); 
    }
    for(; j < mat.m_col; ++j){
      mat[i][j] = 0.0f; 
    }
  } 
}
#else
void mat::mat_ops::zero_mat(){
  for(size_t i = 0; i < mat.m_row; ++i){
    for(size_t j = 0; j < mat.m_col; ++j){
      mat[i][j] = 0; 
    }
  } 
}
#endif 

size_t mat::mat_ops::return_row_count()const{return this->mat.m_row;}
size_t mat::mat_ops::return_col_count()const{return this->mat.m_col;}

#if USE_AVX256
mat::mat_ops mat::mat_ops::mat_mul(const mat_ops &left_mat, const mat_ops &right_mat){
  constexpr int BLOCK_I = 256; //1024 bytes at fp32
  constexpr int BLOCK_J = 256; //1024 bytes at fp32 
  constexpr int BLOCK_K = 16;  //64 bytes at fp32

  const mat::matrix &A = left_mat.mat;
  const mat::matrix &B = right_mat.mat;
  mat::matrix C(A.m_row, B.m_col);

  omp_set_num_threads(omp_get_max_threads());
  #pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for(size_t i_block = 0; i_block < A.m_row; i_block += BLOCK_I) {
    for(size_t j_block = 0; j_block < B.m_col; j_block += BLOCK_J){
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
            //128x128 sized matrices MIN No reason to have it any smaller
            /*If smaller matrices than 128x128 are needed, no reason to use with avx, 
              it's fast enough without for smaller sized ones*/
            size_t j = j_block; 
            for(; j + 63 < j_end; j += 64) {
              __m256 b_vec0 = _mm256_loadu_ps(&B[k][j]);
              __m256 b_vec1 = _mm256_loadu_ps(&B[k][j+8]);
              __m256 b_vec2 = _mm256_loadu_ps(&B[k][j+16]);
              __m256 b_vec3 = _mm256_loadu_ps(&B[k][j+24]); 
              __m256 b_vec4 = _mm256_loadu_ps(&B[k][j+32]);
              __m256 b_vec5 = _mm256_loadu_ps(&B[k][j+40]);
              __m256 b_vec6 = _mm256_loadu_ps(&B[k][j+48]);
              __m256 b_vec7 = _mm256_loadu_ps(&B[k][j+56]); 

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
              for(; j + 7 < j_end; j += 8){
                __m256 b_vec = _mm256_loadu_ps(&B[k][j]); 
                float *c_row = &c_buffer[i - i_block][j - j_block]; 
                __m256 c_vec = _mm256_load_ps(c_row);
                c_vec        = _mm256_fmadd_ps(a_vec, b_vec, c_vec); 
                _mm256_store_ps(c_row, c_vec); 
              }
              for(; j < j_end; ++j){
                c_buffer[i - i_block][j - j_block] += a_val * B[k][j];
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
mat::mat_ops mat::mat_ops::mat_mul(const mat_ops &left_mat, const mat_ops &right_mat){
  size_t mat_size_row = left_mat.mat.m_row;
  size_t mat_size_col = left_mat.mat.m_col; 
  size_t mat_col = right_mat.mat.m_col; 
  mat::matrix temp_mat(mat_size_row, mat_col);
#if DEBUG
  CRUSH_DEBUG("STD MAT_MUL STARTED");
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
mat::mat_ops mat::mat_ops::transpose_matrix(const mat_ops &mat_in){
  mat::matrix res(mat_in.mat.m_col, mat_in.mat.m_row); 
  float block[8][8] __attribute__((aligned(32))); 
  for(size_t i = 0; i < mat_in.mat.m_row; i += 8){
    for(size_t j = 0; j < mat_in.mat.m_col; j += 8){
      for(size_t iblock = 0; iblock < 8; ++iblock){
        _mm256_store_ps(block[iblock], _mm256_load_ps(&mat_in.mat[i+iblock][j]));
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
mat::mat_ops mat::mat_ops::transpose_matrix(const mat_ops &mat_in){
#if DEBUG
  if(mat_in.mat.m_row != mat_in.mat.m_col){
    CRUSH_FATAL("MATRIX IS NOT SQUARE : ASSERTION FAILED"); 
  }
  assert(mat_in.mat.m_row == mat_in.mat.m_col);
#endif
  mat::matrix temp_mat = mat_in.mat; 
  size_t temp = mat_in.mat.m_row; 
  temp_mat.m_row = mat_in.mat.m_col; 
  temp_mat.m_col = temp;
  for(size_t i = 0; i < mat_in.mat.m_row; i++){
    for(size_t j = 0; j < mat_in.mat.m_col; j++){
      temp_mat[i][j] = mat_in.mat[j][i]; 
    }
  }
  return mat_ops(temp_mat);
}
#endif

#if USE_AVX256
mat::mat_ops mat::mat_ops::block_matrix(const mat_ops &mat_in, size_t i, size_t j, size_t p, size_t q){
#if DEBUG
  if(i + p > mat_in.mat.m_row || j + q > mat_in.mat.m_col){
    CRUSH_FATAL("BLOCK INDICES OUT OF RANGE : ASSERTION FALED"); 
  }
  assert(i + p < mat_in.mat.m_row || j + q < mat_in.mat.m_col); 
#endif
  mat::matrix temp_mat(p,q);
  for(size_t a = 0; a < p; ++a){
    for(size_t b= 0; b < q; ++b){
      temp_mat[a][b] = mat_in.mat[a + i][b + j]; 
    }
  }
  return mat_ops(temp_mat); 
}
#else
mat::mat_ops mat::mat_ops::block_matrix(const mat_ops &mat_in, size_t i, size_t j, size_t p, size_t q){
#if DEBUG
  if(i + p > mat_in.mat.m_row || j + q > mat_in.mat.m_col){
    CRUSH_FATAL("BLOCK INDICES OUT OF RANGE : ASSERTION FALED"); 
  }
  assert(i + p < mat_in.mat.m_row || j + q < mat_in.mat.m_col); 
#endif
  mat::matrix temp_mat(p,q);
  for(size_t a = 0; a < p; ++a){
    for(size_t b= 0; b < q; ++b){
      temp_mat[a][b] = mat_in.mat[a + i][b + j]; 
    }
  }
  return mat_ops(temp_mat); 
}
#endif

#if USE_AVX256
mat::mat_ops mat::mat_ops::add_matrix(const mat_ops &left_mat, const mat_ops &right_mat){
#if DEBUG
  if(right_mat.mat.m_row != left_mat.mat.m_col){
    CRUSH_FATAL("MATRIX IS NOT SQUARE : ASSERTION FAILED"); 
  }
  assert(right_mat.mat.m_row == left_mat.mat.m_col);
#endif
  mat::matrix new_mat(right_mat.mat.m_row, right_mat.mat.m_col); 
  mat::mat_ops temp_mat(new_mat); 
 
  for(size_t i = 0; i < right_mat.mat.m_row; ++i){
    for(size_t j = 0; j < right_mat.mat.m_col; ++j){
      temp_mat.mat[i][j] = right_mat.mat[i][j] + left_mat.mat[i][j]; 
    } 
  }
  return temp_mat; 
}
#else
mat::mat_ops mat::mat_ops::add_matrix(const mat_ops &left_mat, const mat_ops &right_mat){
#if DEBUG
  if(right_mat.mat.m_row != left_mat.mat.m_col){
    CRUSH_FATAL("MATRIX IS NOT SQUARE : ASSERTION FAILED"); 
  }
  assert(right_mat.mat.m_row == left_mat.mat.m_col);
#endif
  mat::matrix new_mat(right_mat.mat.m_row, right_mat.mat.m_col); 
  mat::mat_ops temp_mat(new_mat); 
 
  for(size_t i = 0; i < right_mat.mat.m_row; ++i){
    for(size_t j = 0; j < right_mat.mat.m_col; ++j){
      temp_mat.mat[i][j] = right_mat.mat[i][j] + left_mat.mat[i][j]; 
    } 
  }
  return temp_mat; 
}
#endif

