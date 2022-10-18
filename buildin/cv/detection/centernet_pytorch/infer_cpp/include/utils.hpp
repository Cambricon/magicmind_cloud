#ifndef _SAMPLE_UTILS111_HPP
#define _SAMPLE_UTILS111_HPP

#include <fstream>
#include <iostream>
#include "sys/stat.h"
#include <mm_runtime.h>
#include <cnrt.h>

#include <mm_runtime.h>

#define CHECK_CNRT(FUNC, ...) do {                                                                            \
    cnrtRet_t ret = FUNC(__VA_ARGS__);                                                                        \
    if ( ret != CNRT_RET_SUCCESS)                                                                             \
    {  std::cout << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]"  <<  std::endl; \
       abort();                                                                                               \
    }                                                                                                         \
} while(0)

#define CHECK_MM(FUNC, ...) do {                                \
    magicmind::Status ret = FUNC(__VA_ARGS__);                  \
    if ( !ret.ok())                                             \
    {                                                           \
	    std::cout << ret.error_message() << std::endl;      \
    }                                                           \
} while(0)

#define MM_CHECK(status)                                     \
    do                                                       \
    {                                                        \
        auto ret = (status);                                 \
        if (ret != magicmind::Status::OK())                  \
        {                                                    \
            std::cout << "mm failure: " << ret << std::endl; \
            abort();                                         \
        }                                                    \
} while (0)

#define CHECK_PTR(ptr)                               \
    do                                               \
    {                                                \
        if (ptr == nullptr)                          \
        {                                            \
            std::cout << "mm failure " << std::endl; \
            abort();                                 \
        }                                            \
    } while (0)

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

inline bool check_file_exist(std::string path){
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0)
    {
        if ((buffer.st_mode & S_IFDIR) == 0)
            return true;
        return false;
    }
    return false;
}

void PrintModelInfo(magicmind::IModel *model);

// bounding box
struct BBox {
  int label;
  float score;
  int left;
  int top;
  int right;
  int bottom;
};  // struct BBox

#endif // _SAMPLE_UTILS111_HPP