#include "../../include/utils_core/fp16_utils.hpp"

float crush::fp16::half_to_float_scalar(uint16_t half_in) noexcept{
#if defined(__F16C__)
  __m128i half_vec  = _mm_cvtsi32_si128(static_cast<int>(half_in));
  __m128i float_vec = _mm_cvtph_ps(half_vec); 
  return _mm_cvtss_f32(float_vec); 
#else 
  uint16_t half_exponent = (half_in & 0x7C00u);
  uint16_t half_mantissa = (half_in & 0x03FFu); 
  uint16_t signed_bit    = (static_cast<uint32_t>(half_in & 0x8000u) << 16);

  uint32_t float_bits; 
  if(half_exponent == 0){
    if(half_mantissa == 0){
      float_bits = signed_bit; 
    }else{
      int exponent = -14; 
      uint32_t mantissa = half_mantissa; 

      while((mantissa & 0x0400u) == 0){
        mantissa <<= 1; 
        exponent  -= 1; 
      }

      mantissa &= 0x03FFu;

      int      exponent_float = exponent + 127; 
      uint32_t float_mantissa = mantissa << 13;

      float_bits = signed_bit | (static_cast<uint32_t>(exponent_float) << 23) | float_mantissa; 
    }

  }else if(half_exponent == 0x7C00u){
    uint32_t float_mantissa = static_cast<uint32_t>(half_mantissa) << 13; 
    float_bits = signed_bit | 0x7F800000u | float_mantissa; 
  }else{
    int exponent = static_cast<int>(half_exponent >> 10) - 15 + 127; 
    uint32_t float_mantissa = static_cast<uint32_t>(half_mantissa) << 13; 
    float_bits = signed_bit | (static_cast<uint32_t>(exponent) << 23) | float_mantissa; 
  }

  float result; 
  std::memcpy(&result, &float_bits, sizeof(result)); 
  return result; 
#endif
}

uint16_t crush::fp16::float_to_half_scalar(float float_in) noexcept{
#if defined(__F16C__)
  __m128 float_vec = _mm_set_ss(float_in); 
  __m128 half_vec  = _mm_cvtps_ph(float_vec, _MM_FROUD_TO_NEAREST_INT | _MM_FROUND_NO_EXC); 
  return static_cast<uint16_t>(_mm_extract_epi16(half_vec, 0)); 
#else
  uint32_t x_val; 
  std::memcpy(&x_val, &float_in, sizeof(x_val));

  uint32_t signed_bit = (x_val >> 31) & 0x1u; 
  uint32_t exponent   = (x_val >> 23) & 0xFFu; 
  uint32_t mantissa   = (x_val & 0x7FFFFFu); 
  
  uint16_t half_scalar = 0;

  if(exponent == 0xFFu){
    uint16_t half_mantissa = mantissa ? 0x200u | static_cast<uint16_t>(mantissa >> 13) : 0u;
    half_scalar            = static_cast<uint16_t>((signed_bit << 15) | (0x1Fu << 10) | half_mantissa); 
  }else if(exponent <= 112u){
    if(exponent < 103u){
      half_scalar = static_cast<uint16_t>(signed_bit << 15); 
    }else{
      uint32_t mantissa_norm = mantissa | 0x00800000u;

      int shift_offset = static_cast<int>(126u - exponent);
      uint32_t half_mantissa = mantissa_norm  >> (shift_offset + 1);

      if((mantissa_norm >> shift_offset) & 0x1u){
        half_mantissa += 1u; 
      }

      half_scalar = static_cast<uint16_t>((signed_bit << 15) | static_cast<uint16_t>(half_mantissa)); 
    }
  }else if(exponent >= 143u){
    half_scalar = static_cast<uint16_t>((signed_bit << 15) | (0x1Fu << 10));
  }else{
    uint32_t half_exponent = exponent - 112u; 

    uint32_t rounded_mantissa = mantissa + 0x00001000u;
    uint32_t half_mantissa    = rounded_mantissa >> 13;

    if(half_mantissa == 0x400u){
      half_mantissa = 0; 
      half_exponent += 1u; 
      if(half_exponent >= 31u){
        half_exponent = 31u; 
      }
    }
    half_scalar = static_cast<uint16_t>((signed_bit << 15) | (static_cast<uint16_t>(half_exponent) << 10) | static_cast<uint16_t>(half_mantissa)); 
  }
  return half_scalar;
#endif 
}
