#ifndef VECTOR_H  
#define VECTOR_H
#include "crush_defines.h"
#include "logger_core/logger.h"

#define VECTOR
namespace vec{
struct vector{
public:
  size_t m_size;
  __attribute__((aligned(32))) std::vector<float> m_data;
  vector(size_t m_size): m_size(m_size){m_data.reserve(m_size);} 
};//end vector

class vec_ops{
private:
  __attribute__((aligned(32))) vec::vector vec;

public: 
  vec_ops(vec::vector &vec_in): vec(vec_in){}
  
  void display(){
    std::cout<<"{"<<std::endl; 
    for(auto &i : vec.m_data){
      std::cout<<i<<" "; 
    }
    std::cout<<"}"<<std::endl;
  }
  
  void fill_vector(){
    static std::random_device rand; 
    static std::mt19937 gen(rand()); 
    static std::uniform_real_distribution<float> random_value(1,9); 
    for(auto &i : vec.m_data){
      i = random_value(gen);
    }
  }

//start of vector primitives 
//these are different from level1/2/3 BLAS operations
//primitives are still useful in terms of convenience and OOC operations 
#if USE_AVX256
  inline static float fast_reverse_sqrt(float val_in){
  #if DEBUG
    if(val_in < 0){
      CRUSH_FATAL("NEGATIVE NUMBERS ARE NOT ALLOWED : ASSERTION FAILED"); 
    }
    assert(val_in > 0);
  #endif 
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(val_in)));
  }
  inline static float fast_sqrt(float val_in){
  #if DEBUG
    if(val_in < 0){
      CRUSH_FATAL("NEGATIVE NUMBERS ARE NOT ALLOWED : ASSERTION FAILED"); 
    }
    assert(val_in > 0);
  #endif
  return val_in * fast_reverse_sqrt(val_in);
  }
#endif

  static vec_ops add_vectors(const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
  #endif  
    vec::vector temp_vec(left_vec.vec.m_size);
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.m_data[i] = left_vec.vec.m_data[i] + right_vec.vec.m_data[i]; 
    }
  #if DEBUG
    if(temp_vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    return vec_ops(temp_vec);
  }
  
  static void add_vectors(vec_ops &temp_vec, const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
    
    if(temp_vec.vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.vec.m_data[i] = left_vec.vec.m_data[i] + right_vec.vec.m_data[i]; 
    }
  }
  
  static vec_ops sub_vectors(const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
  #endif  
    vec::vector temp_vec(left_vec.vec.m_size);
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.m_data[i] = left_vec.vec.m_data[i] - right_vec.vec.m_data[i]; 
    }
  #if DEBUG
    if(temp_vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    return vec_ops(temp_vec);
  }
  
  static void sub_vectors(vec_ops &temp_vec, const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
    
    if(temp_vec.vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.vec.m_data[i] = left_vec.vec.m_data[i] - right_vec.vec.m_data[i]; 
    }
  }
  
  static vec_ops mul_vectors(const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
  #endif  
    vec::vector temp_vec(left_vec.vec.m_size);
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.m_data[i] = left_vec.vec.m_data[i] * right_vec.vec.m_data[i]; 
    }
  #if DEBUG
    if(temp_vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    return vec_ops(temp_vec);
  }
  
  static void mul_vectors(vec_ops &temp_vec, const vec_ops &left_vec, const vec_ops &right_vec){
  #if DEBUG
    if(left_vec.vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("VECTORS ARE NOT EQUAL IN SIZE : ASSERTION FAILED"); 
    }
    assert(lef_vec.vec.m_size == right_vec.vec.m_size);
    
    if(temp_vec.vec.m_size != left_vec.vec.m_size && temp_vec.m_size != right_vec.vec.m_size){
      CRUSH_FATAL("RESULTING VECTOR IS NOT THE SAME SIZE AS INPUT VECTORS : ASSERTION FAILED"); 
    }
    assert(temp_vec.vec.m_size == left_vec.vec.m_size && temp_vec.m_size == right_vec.vec.m_size);
  #endif
    for(size_t i = 0; i < left_vec.vec.m_size; ++i){
      temp_vec.vec.m_data[i] = left_vec.vec.m_data[i] * right_vec.vec.m_data[i]; 
    }
  }

  static vec_ops scale_vector(const vec_ops &vec_in, float scale){
    vec::vector temp_vec(vec_in.vec.m_size); 
    for(size_t i = 0; i < vec_in.vec.m_size; ++i){
      temp_vec.m_data[i] = scale * vec_in.vec.m_data[i]; 
    }
    return vec_ops(temp_vec);
  }

#if USE_AVX256 
  inline static float get_vector_length(const vec_ops &vec_in){
    return fast_sqrt(std::accumulate(vec_in.vec.m_data.begin(), vec_in.vec.m_data.end(), 0));
  }

  inline static float get_vector_length_squared(const vec_ops &vec_in){
    return std::accumulate(vec_in.vec.m_data.begin(), vec_in.vec.m_data.end(), 0);
  }

  inline static vec_ops normalize_vector(const vec_ops &vec_in){
    const float vec_length = get_vector_length(vec_in);
    if(vec_length != 0.0f){
      return scale_vector(vec_in, (1.0f / vec_length));
    }
    return vec_in;
  }
#else 
  inline static float get_vector_length(const vec_ops &vec_in){
    return std::sqrtf(std::accumulate(vec_in.vec.m_data.begin(), vec_in.vec.m_data.end(), 0)); 
  }

  inline static float get_vector_length_squared(const vec_ops &vec_in){
    return std::accumulate(vec_in.vec.m_data.begin(), vec_in.vec.m_data.end(), 0);
  }

  inline static vec_ops normalize_vector(const vec_ops &vec_in){
    const float vec_length = get_vector_length(vec_in); 
    if(vec_length != 0.0f){
      return scale_vector(vec_in, (1.0f / vec_length)); 
    }
    return vec_in;
  }
#endif
  
};//end vec_ops
}//end namespace
#endif 
