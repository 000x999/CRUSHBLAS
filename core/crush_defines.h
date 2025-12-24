#ifndef DEFINES_H 
#define DEFINES_H
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include <string_view> 
#include <sstream>

#ifdef USE_AVX256
  #include <immintrin.h>
  extern "C" int omp_get_thread_num(); 
  extern "C" int omp_get_num_threads();
  extern "C" int omp_set_dynamic(int threads);
  extern "C" int omp_set_num_threads(int threads);
  extern "C" int omp_get_max_threads();
#endif 

#ifdef DEBUG  
#define DEBUG_THREADS() do {                                \
     _Pragma("omp parallel")                                \
      printf("Thread %d out of %d (File: %s, Line: %d)\n",  \
             omp_get_thread_num(),                          \
             omp_get_num_threads(),                         \
             __FILE__, __LINE__);                           \
  }while(0)
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
  #define CRUSH_PLATFORM_WINDOWS 1
    #ifndef _WIN64 
      #error "64BIT IS REQUIRED ON WINDOWS"
    #endif
  #elif defines(__linux__) || defined(__gnu_linux__)
    #define CRUSH_PLATFORM_LINUX 1
  #endif 

#ifdef CRUSH_EXPORT
  #ifdef _MSC_VER
    #define CRUSH_API __declspec(dllexport)
  #else
    #define CRUSH_API __attribute__((visibility("default")))
  #endif
#else
  #ifdef _MSC_VER
    #define CRUSH_API __declspec(dllimport)
  #else
    #define CRUSH_API
  #endif
#endif

#endif 
