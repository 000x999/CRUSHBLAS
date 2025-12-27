#include "level3/level3.hpp"

namespace{
constexpr size_t max_block_size = 256; 
struct alignas(32) local_buffers{
  float left_pack  [ max_block_size * max_block_size ]; 
  float right_pack [ max_block_size * max_block_size ];
};

thread_local local_buffers buffers; 
}

level3::mat_ops_view::map_view::map_view(float *m_start_row, size_t m_cols)       : m_start_row(m_start_row)                    , m_cols(m_cols){}  
level3::mat_ops_view::map_view::map_view(const float *m_start_row, size_t m_cols) : m_start_row(const_cast<float*>(m_start_row)), m_cols(m_cols){}

float &level3::mat_ops_view::map_view::operator[](size_t col){
#if DEBUG
  if(!(col < m_cols)){
    CRUSH_FATAL("COL INDEX OUT OF BOUNDS :: ASSERTION FAILED"); 
  }
  assert(col < m_cols); 
#endif 
  return m_start_row[col]; 
}

const float &level3::mat_ops_view::map_view::operator[](size_t col)const{
#if DEBUG
  if(!(col < m_cols)){
    CRUSH_FATAL("COL INDEX OOB :: ASSERTION FAILED"); 
  }
  assert(col < m_cols);
#endif   
  return m_start_row[col];     
}

inline float &level3::mat_ops_view::operator()(size_t i, size_t j)       { return data_view[i * leading_dimension + j]; }

const  float &level3::mat_ops_view::operator()(size_t i, size_t j) const { return data_view[i * leading_dimension + j]; } 

inline level3::mat_ops_view::map_view level3::mat_ops_view::operator[](size_t row){
#if DEBUG
  if(!(row_view, row)){
    CRUSH_FATAL("ROW INDEX OOB :: ASSERTION FAILED");
  }
  assert(row_view < row); 
#endif 
  return map_view(&data_view[row * col_view], col_view); 
}

inline const level3::mat_ops_view::map_view level3::mat_ops_view::operator[](size_t row)const{
#if DEBUG
  if(!(row_view, row)){
    CRUSH_FATAL("ROW INDEX OOB :: ASSERTION FAILED");
  }
  assert(row_view < row); 
#endif
  return map_view(&data_view[row * col_view], col_view); 
}

level3::gemm_thread_pool& level3::gemm_thread_pool::get_instance()                                { static gemm_thread_pool instance; return instance; } 
level3::gemm_thread_pool::gemm_thread_pool                      () : active_tasks(0), stop(false) { init_pool(std::thread::hardware_concurrency());    }

void level3::gemm_thread_pool::init_pool                        ( size_t num_threads )            {
  if(num_threads == 0){
    num_threads = 4; 
  }
  for(size_t i = 0; i < num_threads; ++i){
    pool_workers.emplace_back(&gemm_thread_pool::worker_thread, this); 
  }
}

void level3::gemm_thread_pool::shutdown_pool                    ()                                {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true; 
  }
  condition.notify_all();
  for(auto &worker : pool_workers){
    if(worker.joinable()){
      worker.join();
    }
  }
  pool_workers.clear(); 
}

level3::gemm_thread_pool::~gemm_thread_pool                     ()                                { shutdown_pool(); }
void level3::gemm_thread_pool::worker_thread                    ()                                {
  while(true){
    std::function<void()> task; 
    {
      std::unique_lock<std::mutex> lock(queue_mutex); 
      condition.wait(lock, [this]{return stop || !worker_tasks.empty(); });
      if(stop && worker_tasks.empty()){
        return; 
      }

      task = std::move(worker_tasks.front()); 
      worker_tasks.pop(); 
    }

    task();
    {
      std::unique_lock<std::mutex> lock(queue_mutex); 
      --active_tasks; 
    }
    finished.notify_all(); 
  }
}

void level3::gemm_thread_pool::enqueue_task                    (std::function<void()> task_in) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    worker_tasks.push(std::move(task_in));
    ++active_tasks; 
  }
  condition.notify_one(); 
}

void level3::gemm_thread_pool::wait_for_all                    ()                              {
  std::unique_lock<std::mutex> lock(queue_mutex);
  finished.wait(lock, [this]{ return active_tasks == 0 &&  worker_tasks.empty(); }); 
}

#if USE_AVX256 
void level3::blas::gemm(size_t m, size_t n, size_t p, const level3::mat_ops_view &left_view, const level3::mat_ops_view &right_view, float alpha, float beta, level3::mat_ops_view &c_view){
  constexpr int BLOCK_I = 256;  //1024 bytes at fp32
  constexpr int BLOCK_J = 256;  //1024 bytes at fp32
  constexpr int BLOCK_K = 16;   //64 bytes  at fp32
  const mat_ops_view  &A = left_view;
  const mat_ops_view  &B = right_view;
  mat_ops_view  &C = c_view;
#if DEBUG  
  if (A.row_view != m || A.col_view != n) {
    CRUSH_FATAL("GEMM: A dims mismatch"); 
  }
  if (B.row_view != n || B.col_view != p) {
    CRUSH_FATAL("GEMM: B dims mismatch");
  }
  if (C.row_view != m || C.col_view != p) {
    CRUSH_FATAL("GEMM: C dims mismatch");
  }
  if (!A.data_view || !B.data_view || !C.data_view) {
    CRUSH_FATAL("GEMM: null data_view");
  }
#endif
  
  if(p == 1){  
  }else{
    omp_set_num_threads(omp_get_max_threads());
  }
  #pragma omp parallel for collapse(2) schedule(dynamic,1)
  for (size_t i_block = 0; i_block < m; i_block += BLOCK_I) {
    for (size_t j_block = 0; j_block < p; j_block += BLOCK_J) {

      float c_buffer[BLOCK_I][BLOCK_J] __attribute__((aligned(32))) = {{0}};
      const size_t i_end = std::min(i_block + BLOCK_I, m);
      const size_t j_end = std::min(j_block + BLOCK_J, p);

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
      for (size_t k_block = 0; k_block < n; k_block += BLOCK_K) {
        const size_t k_end = std::min(k_block + BLOCK_K, n);
        
        for (size_t i = i_block; i < i_end; ++i) {
          for (size_t k = k_block; k < k_end; ++k) {  
            float a_val = A[i][k];
            __m256 alpha_vec = _mm256_set1_ps(alpha);  
            __m256 a_vec     = _mm256_broadcast_ss(&a_val);
            if (alpha != 1.0f) {
              a_vec = _mm256_mul_ps(alpha_vec, a_vec);
            }
            size_t j = j_block;
#if DEBUG                 
            if (A.row_view != m || A.col_view != n) {
              CRUSH_FATAL("GEMM: A dims mismatch"); 
            }
            if (B.row_view != n || B.col_view != p) {
              CRUSH_FATAL("GEMM: B dims mismatch");
            }
            if (C.row_view != m || C.col_view != p) {
              CRUSH_FATAL("GEMM: C dims mismatch");
            }
            if (!A.data_view || !B.data_view || !C.data_view) {
              CRUSH_FATAL("GEMM: null data_view");
            }
#endif 
            //8x unrolled loop
            //128x128 sized matrices MIN No reason to have it any smaller
            /*If smaller matrices than 128x128 are needed, no reason to use with avx, 
              it's fast enough without for smaller sized ones*/
          //std::cout << "unrolled loop start" << '\n'; 
            for (; j + 67 < j_end; j += 64) {
              __m256 b_vec0 = _mm256_loadu_ps(&B[k][j]);
              __m256 b_vec1 = _mm256_loadu_ps(&B[k][j +  8]);
              __m256 b_vec2 = _mm256_loadu_ps(&B[k][j + 16]);
              __m256 b_vec3 = _mm256_loadu_ps(&B[k][j + 24]);
              __m256 b_vec4 = _mm256_loadu_ps(&B[k][j + 32]);
              __m256 b_vec5 = _mm256_loadu_ps(&B[k][j + 40]);
              __m256 b_vec6 = _mm256_loadu_ps(&B[k][j + 48]);
              __m256 b_vec7 = _mm256_loadu_ps(&B[k][j + 56]);

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
            //std::cout << "handling excess from unrolled loop" << '\n'; 
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
          //std::cout << "C[i][j]" << C[i][j] <<  " = " << "c_buffer[i - i_block][j - j_block]" << c_buffer[i - i_block][j - j_block] << '\n'; 
          C[i][j] = c_buffer[i - i_block][j - j_block];
        }
      }
    }
  }
#if DEBUG
  DEBUG_THREADS();
#endif
}
#endif

void level3::blas::pack_left_block(transpose_gemm transpose_left, const mat_ops_view &left_view, size_t index_zero, size_t kindex_zero, size_t m_c, size_t k_c, float *left_pack){
  const size_t leading_dimension_left = left_view.leading_dimension; 
  if(transpose_left == transpose_gemm::no_transpose){
    for(size_t i = 0; i < m_c; ++i){
      const float *source_data        = left_view.data_view + (index_zero + i) * leading_dimension_left + kindex_zero; 
      float       *destination_block  = left_pack + i * k_c; 
      std::memcpy(destination_block, source_data, k_c * sizeof(float)); 
    }
  }else{
    for(size_t i = 0; i < m_c; ++i){
      float *destination_block = left_pack + i * k_c; 
      size_t global_index      = index_zero + i;
      for(size_t k = 0; k < k_c; ++k){
        size_t global_kindex   = kindex_zero + k; 
        destination_block[k] = left_view.data_view[global_kindex * leading_dimension_left + global_index]; 
      }
    }
  }
}

void level3::blas::pack_right_block(transpose_gemm transpose_right, const mat_ops_view &right_view, size_t kindex_zero, size_t jindex_zero, size_t k_c, size_t n_c, float *right_pack){
  const size_t leading_dimension_right = right_view.leading_dimension; 

  if(transpose_right == transpose_gemm::no_transpose){
    for(size_t k = 0; k < k_c; ++k){
      const float *source_data       = right_view.data_view + (kindex_zero + k) * leading_dimension_right + jindex_zero; 
      float       *destination_block = right_pack + k * n_c;
      std::memcpy(destination_block, source_data, n_c * sizeof(float)); 
    }

  }else{
    for(size_t k = 0; k < k_c; ++k){
      float *destination_block = right_pack + k * n_c; 
      size_t row_op            = kindex_zero + k; 
      
      for(size_t j = 0; j < n_c; ++j){
        size_t col_op          = jindex_zero + j; 
        size_t row_right       = col_op;
        size_t col_right       = row_op;
        destination_block[j]   = right_view.data_view[row_right * leading_dimension_right + col_right]; 
      }
    }
  }
}

void level3::blas::microkernel_4x8_avx256(const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta){
  __m256 c_vec_zero  = _mm256_setzero_ps();
  __m256 c_vec_one   = _mm256_setzero_ps();
  __m256 c_vec_two   = _mm256_setzero_ps();
  __m256 c_vec_three = _mm256_setzero_ps();
  
  for(size_t k = 0; k < k_c; ++k){
    const float *right_row_data   = right_block + k * leading_dimension_right;
    __m256 right_block_vec        = _mm256_loadu_ps(right_row_data);

    float left_row_element_zero   = left_block[0 * leading_dimension_left + k]; 
    float left_row_element_one    = left_block[1 * leading_dimension_left + k];
    float left_row_element_two    = left_block[2 * leading_dimension_left + k];
    float left_row_element_three  = left_block[3 * leading_dimension_left + k];
 
    __m256 left_vec_zero          = _mm256_set1_ps(left_row_element_zero); 
    __m256 left_vec_one           = _mm256_set1_ps(left_row_element_one); 
    __m256 left_vec_two           = _mm256_set1_ps(left_row_element_two);
    __m256 left_vec_three         = _mm256_set1_ps(left_row_element_three);

    c_vec_zero                    = _mm256_fmadd_ps(left_vec_zero , right_block_vec, c_vec_zero);  
    c_vec_one                     = _mm256_fmadd_ps(left_vec_one  , right_block_vec, c_vec_one); 
    c_vec_two                     = _mm256_fmadd_ps(left_vec_two  , right_block_vec, c_vec_two);
    c_vec_three                   = _mm256_fmadd_ps(left_vec_three, right_block_vec, c_vec_three);
  }
 
  if(beta != 0.0f || alpha != 1.0f){
    __m256 beta_vec        = _mm256_set1_ps(beta);  
    __m256 alpha_vec       = _mm256_set1_ps(alpha);

    __m256 c_vec_zero_old  = _mm256_loadu_ps(c_block + 0 * leading_dimension_c); 
    __m256 c_vec_one_old   = _mm256_loadu_ps(c_block + 1 * leading_dimension_c);
    __m256 c_vec_two_old   = _mm256_loadu_ps(c_block + 2 * leading_dimension_c);
    __m256 c_vec_three_old = _mm256_loadu_ps(c_block + 3 * leading_dimension_c);

    c_vec_zero             = _mm256_fmadd_ps(c_vec_zero_old , beta_vec, _mm256_mul_ps(alpha_vec, c_vec_zero)); 
    c_vec_one              = _mm256_fmadd_ps(c_vec_one_old  , beta_vec, _mm256_mul_ps(alpha_vec, c_vec_one));
    c_vec_two              = _mm256_fmadd_ps(c_vec_two_old  , beta_vec, _mm256_mul_ps(alpha_vec, c_vec_two));
    c_vec_three            = _mm256_fmadd_ps(c_vec_three_old, beta_vec, _mm256_mul_ps(alpha_vec, c_vec_three));
  }

  _mm256_storeu_ps(c_block + 0 * leading_dimension_c, c_vec_zero);
  _mm256_storeu_ps(c_block + 1 * leading_dimension_c, c_vec_one);
  _mm256_storeu_ps(c_block + 2 * leading_dimension_c, c_vec_two);
  _mm256_storeu_ps(c_block + 3 * leading_dimension_c, c_vec_three);
}

void level3::blas::microkernel_6x16_avx256(const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta){
  __m256 reg_00 = _mm256_setzero_ps(); 
  __m256 reg_10 = _mm256_setzero_ps();
  __m256 reg_20 = _mm256_setzero_ps();
  __m256 reg_30 = _mm256_setzero_ps();
  __m256 reg_40 = _mm256_setzero_ps();
  __m256 reg_50 = _mm256_setzero_ps();
  
  __m256 reg_01 = _mm256_setzero_ps();
  __m256 reg_11 = _mm256_setzero_ps();
  __m256 reg_21 = _mm256_setzero_ps();
  __m256 reg_31 = _mm256_setzero_ps();
  __m256 reg_41 = _mm256_setzero_ps();
  __m256 reg_51 = _mm256_setzero_ps();

  for(size_t k = 0; k < k_c; ++k){
    const float *right_row = right_block + k * leading_dimension_right; 
    __m256 reg_b0 = _mm256_loadu_ps(right_row);
    __m256 reg_b1 = _mm256_loadu_ps(right_row + 8); 

    __m256 reg_a0 = _mm256_set1_ps(left_block[0 * leading_dimension_left + k]); 
    __m256 reg_a1 = _mm256_set1_ps(left_block[1 * leading_dimension_left + k]);
    __m256 reg_a2 = _mm256_set1_ps(left_block[2 * leading_dimension_left + k]);
    __m256 reg_a3 = _mm256_set1_ps(left_block[3 * leading_dimension_left + k]);
    __m256 reg_a4 = _mm256_set1_ps(left_block[4 * leading_dimension_left + k]);
    __m256 reg_a5 = _mm256_set1_ps(left_block[5 * leading_dimension_left + k]);
  
    reg_00 = _mm256_fmadd_ps(reg_a0, reg_b0, reg_00); 
    reg_10 = _mm256_fmadd_ps(reg_a1, reg_b0, reg_10);
    reg_20 = _mm256_fmadd_ps(reg_a2, reg_b0, reg_20);
    reg_30 = _mm256_fmadd_ps(reg_a3, reg_b0, reg_30);
    reg_40 = _mm256_fmadd_ps(reg_a4, reg_b0, reg_40);
    reg_50 = _mm256_fmadd_ps(reg_a5, reg_b0, reg_50);

    reg_01 = _mm256_fmadd_ps(reg_a0, reg_b1, reg_01);
    reg_11 = _mm256_fmadd_ps(reg_a1, reg_b1, reg_11);
    reg_21 = _mm256_fmadd_ps(reg_a2, reg_b1, reg_21);
    reg_31 = _mm256_fmadd_ps(reg_a3, reg_b1, reg_31);
    reg_41 = _mm256_fmadd_ps(reg_a4, reg_b1, reg_41);
    reg_51 = _mm256_fmadd_ps(reg_a5, reg_b1, reg_51);  
  }
  
  if(beta != 0.0f || alpha != 1.0f){
    __m256 alpha_vec = _mm256_set1_ps(alpha); 
    __m256 beta_vec  = _mm256_set1_ps(beta); 

    #define scale_and_store(row, reg_0, reg_1)\
      {\
        __m256 old_reg_0 = _mm256_loadu_ps(c_block  + row * leading_dimension_c);\
        __m256 old_reg_1 = _mm256_loadu_ps(c_block  + row * leading_dimension_c + 8);\
        reg_0 = _mm256_fmadd_ps(old_reg_0, beta_vec, _mm256_mul_ps(alpha_vec, reg_0));\
        reg_1 = _mm256_fmadd_ps(old_reg_1, beta_vec, _mm256_mul_ps(alpha_vec, reg_1));\
        _mm256_storeu_ps(c_block + row * leading_dimension_c, reg_0);\
        _mm256_storeu_ps(c_block + row * leading_dimension_c + 8, reg_1);\
      }
      scale_and_store(0, reg_00, reg_01); 
      scale_and_store(1, reg_10, reg_11);
      scale_and_store(2, reg_20, reg_21);
      scale_and_store(3, reg_30, reg_31);
      scale_and_store(4, reg_40, reg_41);
      scale_and_store(5, reg_50, reg_51);
    #undef scale_and_store
  }else{
    _mm256_storeu_ps(c_block + 0 * leading_dimension_c,     reg_00); 
    _mm256_storeu_ps(c_block + 0 * leading_dimension_c + 8, reg_01);
    _mm256_storeu_ps(c_block + 1 * leading_dimension_c,     reg_10);
    _mm256_storeu_ps(c_block + 1 * leading_dimension_c + 8, reg_11);
    _mm256_storeu_ps(c_block + 2 * leading_dimension_c,     reg_20);
    _mm256_storeu_ps(c_block + 2 * leading_dimension_c + 8, reg_21);
    _mm256_storeu_ps(c_block + 3 * leading_dimension_c,     reg_30);
    _mm256_storeu_ps(c_block + 3 * leading_dimension_c + 8, reg_31);
    _mm256_storeu_ps(c_block + 4 * leading_dimension_c,     reg_40);
    _mm256_storeu_ps(c_block + 4 * leading_dimension_c + 8, reg_41);
    _mm256_storeu_ps(c_block + 5 * leading_dimension_c,     reg_50);
    _mm256_storeu_ps(c_block + 5 * leading_dimension_c + 8, reg_51); 
  }
}

void level3::blas::microkernel_eno_edge(size_t m_r, size_t n_r, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t k_c, float alpha, float beta){
  for(size_t i = 0; i < m_r; ++i){
    for(size_t j = 0; j < n_r; ++j){
      float edge_sum = 0.0f; 
      for(size_t k = 0; k < k_c; ++k){
        float left_edge  = left_block[i * leading_dimension_left + k]; 
        float right_edge = right_block[k * leading_dimension_right +j]; 
        edge_sum += left_edge * right_edge; 
      }
      float &c_block_ij = c_block[i * leading_dimension_c + j]; 
      if(beta == 0.0f){
        c_block_ij = alpha * edge_sum; 
      }else{
        c_block_ij = alpha * edge_sum + beta * c_block_ij;
      }
    }
  } 
}

void level3::blas::gemm_avx256_core(transpose_gemm transpose_left, transpose_gemm transpose_right, size_t m, size_t n, size_t k, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, size_t i_begin, size_t i_end, float alpha, float beta){
  assert(left_block && right_block && c_block);
  assert(i_begin <= i_end && i_end <= m);
  mat_ops_view left_view  { transpose_left == transpose_gemm::no_transpose ? m : k, transpose_left == transpose_gemm::no_transpose ? k : m, leading_dimension_left, const_cast<float*>(left_block) }; 
  mat_ops_view right_view { transpose_right == transpose_gemm::no_transpose ? k : n, transpose_right == transpose_gemm::no_transpose ? n : k, leading_dimension_right, const_cast<float*>(right_block) };
  mat_ops_view c_view     { m, n, leading_dimension_c, c_block };
  constexpr size_t m_r = 4; 
  constexpr size_t n_r = 8; 

  constexpr size_t m_block = 256; 
  constexpr size_t n_block = 256; 
  constexpr size_t k_block = 256; 

  float *left_pack_data  = buffers.left_pack; 
  float *right_pack_data = buffers.right_pack;  
  
  for(size_t i_zero = i_begin; i_zero < i_end; i_zero += m_block){
    const size_t m_c = std::min(m_block, i_end - i_zero); 
    
    for(size_t k_zero = 0; k_zero < k; k_zero += k_block){
      const size_t k_c = std::min(k_block, k - k_zero);

      level3::blas::pack_left_block(transpose_left, left_view, i_zero, k_zero, m_c, k_c, left_pack_data);
      
      for(size_t j_zero = 0; j_zero < n; j_zero += n_block){
        const size_t n_c = std::min(n_block, n - j_zero);

        level3::blas::pack_right_block(transpose_right, right_view, k_zero, j_zero, k_c, n_c, right_pack_data); 
        const bool  first_k_block = (k_zero == 0);
        const float eno_eff      = first_k_block ? beta : 1.0f; 

        for(size_t index_one = 0; index_one < m_c; index_one += m_r){
          const size_t m_r_diff = std::min(m_r, m_c - index_one); 

          for(size_t jindex_one = 0; jindex_one < n_c; jindex_one += n_r){
            const  size_t n_r_diff = std::min(n_r, n_c - jindex_one);

            float       *c_data_tile = c_view.data_view + (i_zero + index_one) * leading_dimension_c + (j_zero + jindex_one);
            const float *left_tile   = left_pack_data + index_one * k_c; 
            const float *right_tile  = right_pack_data + jindex_one;

            if(m_r_diff == m_r && n_r_diff == n_r){
              level3::blas::microkernel_6x16_avx256(left_tile, k_c, right_tile, n_c, c_data_tile, leading_dimension_c, k_c, alpha, eno_eff); 
            }else{
              level3::blas::microkernel_eno_edge(m_r_diff, n_r_diff, left_tile, k_c, right_tile, n_c, c_data_tile, leading_dimension_c, k_c, alpha, eno_eff);
            }
          }
        }
      }
    }
  }
}

void level3::blas::gemm_avx256_mt(transpose_gemm transpose_left, transpose_gemm transpose_right, size_t m, size_t n, size_t k, const float *left_block, size_t leading_dimension_left, const float *right_block, size_t leading_dimension_right, float *c_block, size_t leading_dimension_c, float alpha, float beta){
  assert(left_block && right_block && c_block);
  auto &pool = gemm_thread_pool::get_instance();
  
  const unsigned hardware_count = std::thread::hardware_concurrency(); 
  const unsigned max_threads    = (hardware_count == 0 ? 4 : hardware_count);

  const size_t   min_rows_per_thread = 48; 
  const unsigned threads_per_rows    = static_cast<unsigned>(std::max<size_t>(1, m / min_rows_per_thread));
  const unsigned num_threads         = std::min(max_threads, threads_per_rows);

  if(num_threads <= 1){
    level3::blas::gemm_avx256_core(
    transpose_left, transpose_right, m, n, k,
    left_block    , leading_dimension_left,
    right_block   , leading_dimension_right,
    c_block       , leading_dimension_c, 
    0             , m, alpha, beta
    );
    return; 
  }

  const size_t rows_per_thread = ((m + num_threads - 1) / num_threads + 5) / 6 * 6;

  for(unsigned thread_index = 0; thread_index < num_threads; ++thread_index){
    const size_t i_begin = thread_index * rows_per_thread; 
    if(i_begin >= m){
      break;
    }
    const size_t i_end = std::min(m, i_begin + rows_per_thread); 

    pool.enqueue_task([=](){
      level3::blas::gemm_avx256_core(
      transpose_left, transpose_right, m, n, k,
      left_block    , leading_dimension_left, 
      right_block   , leading_dimension_right, 
      c_block       , leading_dimension_c, 
      i_begin       , i_end, alpha, beta
      ); 
    });
  } 

  pool.wait_for_all(); 
}
#if USE_BLAS
void level3::blas::crush_gemm(transpose_gemm transpose_left, transpose_gemm transpose_right, const mat_ops_view &left_view, const mat_ops_view &right_view, float alpha, float beta, mat_ops_view &c_view){
  size_t m = (transpose_left  == transpose_gemm::no_transpose) ? left_view.row_view  : left_view.col_view; 
  size_t k = (transpose_left  == transpose_gemm::no_transpose) ? left_view.col_view  : left_view.row_view;
  size_t n = (transpose_right == transpose_gemm::no_transpose) ? right_view.col_view : right_view.row_view;
  
#if DEBUG
  size_t k_b = (transpose_right == transpose_gemm::no_transpose) ? right_view.row_view : right_view.col_view; 
  assert(k_b == k);
  assert(c_view.row_view == m && c_view.col_view == n); 
#endif 
  
  size_t leading_dimension_left  = left_view.leading_dimension; 
  size_t leading_dimension_right = right_view.leading_dimension; 
  size_t leading_dimension_c     = c_view.leading_dimension; 
  level3::blas::gemm_avx256_mt(transpose_left, transpose_right, m, n, k, left_view.data_view, leading_dimension_left, right_view.data_view, leading_dimension_right, c_view.data_view, leading_dimension_c, alpha, beta); 
}
#endif 

#if USE_AVX256
void level3::blas::crush_gemm(transpose_gemm transpose_left, transpose_gemm transpose_right, const mat_ops_view &left_view, const mat_ops_view &right_view, float alpha, float beta, mat_ops_view &c_view){
  size_t m = (transpose_left  == transpose_gemm::no_transpose) ? left_view.row_view  : left_view.col_view;
  size_t k = (transpose_left  == transpose_gemm::no_transpose) ? left_view.col_view  : left_view.row_view;
  size_t n = (transpose_right == transpose_gemm::no_transpose) ? right_view.col_view : right_view.row_view;

  constexpr size_t BLOCK_I = 16;
  constexpr size_t BLOCK_J = 256;
  constexpr size_t BLOCK_K = 2;

  float* A = left_view.data_view;
  float* B = right_view.data_view;
  float* C = c_view.data_view;
  
  size_t lda = left_view.leading_dimension;
  size_t ldb = right_view.leading_dimension;
  size_t ldc = c_view.leading_dimension;

  omp_set_num_threads(omp_get_max_threads());
  
  #pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (size_t i_block = 0; i_block < m; i_block += BLOCK_I) {
    for (size_t j_block = 0; j_block < n; j_block += BLOCK_J) {
      alignas(32) float c_buffer[BLOCK_I][BLOCK_J] = {{0}};
      
      size_t i_end = std::min(i_block + BLOCK_I, m);
      size_t j_end = std::min(j_block + BLOCK_J, n);

      if (beta != 0.0f) {
        for (size_t ii = i_block; ii < i_end; ++ii) {
          for (size_t jj = j_block; jj < j_end; ++jj) {
            c_buffer[ii - i_block][jj - j_block] = beta * C[ii * ldc + jj];
            }
          }
      }
      for (size_t k_block = 0; k_block < k; k_block += BLOCK_K) {
        size_t k_end = std::min(k_block + BLOCK_K, k);

        for (size_t i = i_block; i < i_end; ++i) {
          for (size_t kk = k_block; kk < k_end; ++kk) {
            float a_val = A[i * lda + kk];
            __m512 a_vec = _mm512_set1_ps(a_val * alpha);

            size_t j = j_block;
              
            for (; j + 63 < j_end; j += 64) {
              __m512 b0 = _mm512_loadu_ps(&B[kk * ldb + j]);
              __m512 b1 = _mm512_loadu_ps(&B[kk * ldb + j + 8]);
              __m512 b2 = _mm512_loadu_ps(&B[kk * ldb + j + 16]);
              __m512 b3 = _mm512_loadu_ps(&B[kk * ldb + j + 24]);
              __m512 b4 = _mm512_loadu_ps(&B[kk * ldb + j + 32]);
              __m512 b5 = _mm512_loadu_ps(&B[kk * ldb + j + 40]);
              __m512 b6 = _mm512_loadu_ps(&B[kk * ldb + j + 48]);
              __m512 b7 = _mm512_loadu_ps(&B[kk * ldb + j + 56]);

              __m512 c0 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block]);
              __m512 c1 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 8]);
              __m512 c2 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 16]);
              __m512 c3 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 24]);
              __m512 c4 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 32]);
              __m512 c5 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 40]);
              __m512 c6 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 48]);
              __m512 c7 = _mm512_load_ps(&c_buffer[i - i_block][j - j_block + 56]);

              c0 = _mm512_fmadd_ps(a_vec, b0, c0);
              c1 = _mm512_fmadd_ps(a_vec, b1, c1);
              c2 = _mm512_fmadd_ps(a_vec, b2, c2);
              c3 = _mm512_fmadd_ps(a_vec, b3, c3);
              c4 = _mm512_fmadd_ps(a_vec, b4, c4);
              c5 = _mm512_fmadd_ps(a_vec, b5, c5);
              c6 = _mm512_fmadd_ps(a_vec, b6, c6);
              c7 = _mm512_fmadd_ps(a_vec, b7, c7);

              _mm512_store_ps(&c_buffer[i - i_block][j - j_block], c0);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 8], c1);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 16], c2);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 24], c3);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 32], c4);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 40], c5);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 48], c6);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block + 56], c7);
            }

            for (; j + 15 < j_end; j += 16) {
              __m512 b_vec = _mm512_loadu_ps(&B[kk * ldb + j]);
              __m512 c_vec = _mm512_load_ps(&c_buffer[i - i_block][j - j_block]);
              c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
              _mm512_store_ps(&c_buffer[i - i_block][j - j_block], c_vec);
            }

            for (; j < j_end; ++j) {
              c_buffer[i - i_block][j - j_block] += a_val * alpha * B[kk * ldb + j];
            }
          }
        }
      }
      for (size_t i = i_block; i < i_end; ++i) {
        for (size_t j = j_block; j < j_end; ++j) {
          C[i * ldc + j] = c_buffer[i - i_block][j - j_block];
        }
      }
    }
  }
}
#endif 

level3::mat_ops_view level3::blas::softmax(level3::mat_ops_view &input_view){
  size_t row = input_view.row_view;
  size_t col = input_view.col_view; 
  
  for(size_t i = 0; i < row; ++i){
    float *row_ptr = input_view.data_view + i * input_view.leading_dimension; 

    float max_val = row_ptr[0];
    for(size_t j = 1; j < col; ++j){
      if(row_ptr[j] > max_val){
        max_val = row_ptr[j]; 
      }
    }

    float curr_sum = 0.0f;
    for(size_t j = 0; j < col; ++j){
      row_ptr[j] = std::exp(row_ptr[j] - max_val);
      curr_sum += row_ptr[j]; 
    }

    for(size_t j = 0; j < col; ++j){
      row_ptr[j] /= curr_sum; 
    }
  }
  return input_view;
}
