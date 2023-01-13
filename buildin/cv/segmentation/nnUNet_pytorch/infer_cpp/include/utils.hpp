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

template <typename T>
std::vector<T> ReadBinFile(const std::string &path) {
  int size = 0;
  std::ifstream ifs(path, std::ifstream::binary);
  if (!ifs.is_open()) { std::cout << "Open label file failed. path : " << path << std::endl;}
  ifs.seekg(0, ifs.end);
  size = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::vector<T> data(size / sizeof(T));
  ifs.read(reinterpret_cast<char *>(data.data()), size);
  ifs.close();
  return data;
}

template <typename T>
void WriteBinFile(const std::string &path, int count, T *ptr) {
  std::ofstream ofresult(path);
  int size = count * sizeof(T);
  ofresult.write(reinterpret_cast<char *>(ptr), size);
  ofresult.close();
}


inline static void PrintModelInfo(magicmind::IModel *model) {
  std::cout << "==================Model info===================" << std::endl;
  std::cout << "Input number : " << model->GetInputNum() << std::endl;
  for (int i = 0; i < model->GetInputNum(); ++i)
    std::cout << "input[" << i << "] : dimensions "
              << model->GetInputDimension(i) << ", data type ["
              << model->GetInputDataType(i) << "]" << std::endl;
  std::cout << "Output number : " << model->GetOutputNum() << std::endl;
  for (int i = 0; i < model->GetOutputNum(); ++i)
    std::cout << "output[" << i << "] : dimensions "
              << model->GetOutputDimension(i) << ", data type ["
              << model->GetOutputDataType(i) << "]" << std::endl;
}
#endif // _SAMPLE_UTILS111_HPP

