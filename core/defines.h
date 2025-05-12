#ifndef DEFINES_H 
#define DEFINES_H

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#define CRUSH_PLATFORM_WINDOWS 1
#ifndef _WIN64 
#error "64-Bit is required on Windows"
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
#ifdef _MSC_VER
#define CRUSH_API __declspec(dllimport)
#else 
#define CRUSH_API
#endif
#endif



#endif 
