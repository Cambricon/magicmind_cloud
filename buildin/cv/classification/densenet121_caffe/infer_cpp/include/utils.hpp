#ifndef UTILS_HPP
#define UTILS_HPP
#include <cnrt.h>
#include <fstream>
#include <iostream>
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

#define MM_CHECK(status) do {                                                                     \
    auto ret = (status);                                                                          \
    if (ret != magicmind::Status::OK())                                                           \
    {                                                                                             \
        std::cout << "[" << __FILE__ << ":" << __LINE__ << "]  mm failure: " << ret << std::endl; \
        abort();                                                                                  \
    }                                                                                             \
} while (0)

#define CHECK_PTR(ptr) do {                      \
    if (ptr == nullptr)                          \
    {                                            \
        std::cout << "mm failure " << std::endl; \
        abort();                                 \
    }                                            \
} while (0)

class Record
{
public:
    Record(std::string filename){
        outfile.open((filename).c_str(), std::ios::trunc | std::ios::out);
    }

    ~Record(){
        if(outfile.is_open())
            outfile.close();
    }

    void write(std::string line, bool print = false){
        outfile << line << std::endl;
        if (print)
        {
            std::cout << line << std::endl;
        }
    }

private:
    std::ofstream outfile;
};

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

void PrintModelInfo(magicmind::IModel *model);
bool CheckModel(magicmind::IModel *model);
#endif //UTILS_HPP

