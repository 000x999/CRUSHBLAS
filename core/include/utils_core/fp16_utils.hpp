#ifndef FP16_UTILS_H 
#define FP16_UTILS_H 

#include <cstdint> 
#include <cstring> 
#include <cmath> 
#include <immintrin.h> 

namespace crush{
class fp16{

public: 
static float         half_to_float_scalar (uint16_t half_in) noexcept;
static uint16_t      float_to_half_scalar (float float_in)   noexcept; 

};//class
}; //namespace

#endif 
