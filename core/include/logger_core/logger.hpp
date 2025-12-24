#ifndef LOGGER_H
#define LOGGER_H 
#include "../../defines.h"

#define CRUSH_LOG_WARN_ENABLED  1
#define CRUSH_LOG_INFO_ENABLED  1
#define CRUSH_LOG_DEBUG_ENABLED 1

typedef enum crush_log_level { 
  CRUSH_LOG_LEVEL_FATAL = 0, 
  CRUSH_LOG_LEVEL_ERROR = 1,
  CRUSH_LOG_LEVEL_WARN  = 2, 
  CRUSH_LOG_LEVEL_INFO  = 3, 
  CRUSH_LOG_LEVEL_DEBUG = 4,  
}crush_log_level;  

bool init_logging(); 
bool suspend_logging(); 

template <typename... Args>
inline void log_output(crush_log_level log_level, Args&&... args){
  static constexpr const char *log_level_strings[] = {"\033[31m[FATAL]", "\033[31m[ERROR]", "\033[33m[WARN]","\033[32m[INFO]", "[DEBUG]"};
  bool is_error = log_level < 2; 
  std::ostringstream log_string_stream; 
  log_string_stream << log_level_strings[log_level] << " ";
  (void)(log_string_stream << ... << std::forward<Args>(args));
  auto log_string_buffer = log_string_stream.str(); 
  FILE *log_message_out = is_error ? stderr : stdout;
  std::fwrite(log_string_buffer.c_str(), 1, log_string_buffer.size(), log_message_out); 
}
template void log_output<>(crush_log_level); 
#ifndef CRUSH_FATAL
  #define CRUSH_FATAL(...)\
  do{\
    log_output(CRUSH_LOG_LEVEL_FATAL, __VA_ARGS__);\
  }while(0)
#endif

#ifndef CRUSH_ERROR
  #define CRUSH_ERROR(...)\
  do{\
    log_output(CRUSH_LOG_LEVEL_ERROR, __VA_ARGS__);\
  }while(0)
#endif 

#if CRUSH_lOG_WARN_ENABLED == 1
  #define CRUSH_WARN(...)\
  do{\
    log_output(CRUSH_LOG_LEVEL_WARN, __VA_ARGS__);\
  }while(0)
#else 
  #define CRUSH_WARN(...)
#endif 

#if CRUSH_LOG_INFO_ENABLED == 1
  #define CRUSH_INFO(...)\
  do{\
    log_output(CRUSH_LOG_LEVEL_INFO, __VA_ARGS__);\
  }while(0)
#else 
  #define CRUSH_INFO(...)
#endif 

#if CRUSH_LOG_DEBUG_ENABLED == 1
  #define CRUSH_DEBUG(...)\
  do{\
    log_output(CRUSH_LOG_LEVEL_DEBUG, __VA_ARGS__);\
  }while(0)
#else 
  #define CRUSH_DEBUG(...)
#endif

#endif 
