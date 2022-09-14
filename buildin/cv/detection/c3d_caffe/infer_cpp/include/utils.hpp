#ifndef UTILS_H_
#define UTILS_H_

#include <cnrt.h>
#include <mm_runtime.h>
#include <glog/logging.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "post_process.hpp"

#define CHECK_CNRT(FUNC, ...) do {                                                 \
  cnrtRet_t ret = FUNC(__VA_ARGS__);                                               \
  LOG_IF(FATAL, CNRT_RET_SUCCESS != ret)                                           \
    << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]";  \
} while(0)

#define MM_CHECK(status)                                                                          \
do                                                                                                \
{                                                                                                 \
    auto ret = (status);                                                                          \
    if (ret != magicmind::Status::OK())                                                           \
    {                                                                                             \
        std::cout << "[" << __FILE__ << ":" << __LINE__ << "]  mm failure: " << ret << std::endl; \
        abort();                                                                                  \
    }                                                                                             \
} while (0)

#define CHECK_VALID(VALUE) do {           \
  auto t = (VALUE);                       \
  LOG_IF(FATAL, !t)                       \
    << #VALUE << " check valid failed.";  \
} while(0)

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

void PrintModelInfo(magicmind::IModel *model) ;
bool CheckModel(magicmind::IModel *model);

#endif  // UTILS_H_
