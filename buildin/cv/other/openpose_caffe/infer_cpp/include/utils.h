#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include <mm_runtime.h>
#include <glog/logging.h>
#include <cnrt.h>
#include <vector>
#include <regex>
#include "sys/stat.h"
#include "net_params.h"


#define CHECK_CNRT(FUNC, ...)                                                         \
  do                                                                                  \
  {                                                                                   \
    cnrtRet_t ret = FUNC(__VA_ARGS__);                                                \
    LOG_IF(FATAL, CNRT_RET_SUCCESS != ret)                                            \
        << "Call " << #FUNC << " failed. Ret code [" << static_cast<int>(ret) << "]"; \
  } while (0)

#define CHECK_PTR(ptr)                         \
  do                                           \
  {                                            \
    if (ptr == nullptr)                        \
    {                                          \
      LOG(INFO) << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)

#define CHECK_MM(FUNC, ...) do {                                \
    magicmind::Status ret = FUNC(__VA_ARGS__);                  \
    if ( !ret.ok())                                             \
    {                                                           \
	    std::cout << ret.error_message() << std::endl;      \
    }                                                           \
} while(0)

#define CHECK_STATUS(status)                               \
  do                                                   \
  {                                                    \
    auto ret = (status);                               \
    if (ret != magicmind::Status::OK())                \
    {                                                  \
      LOG(INFO) << "mm failure: " << ret << std::endl; \
      abort();                                         \
    }                                                  \
  } while (0)

class MluDeviceGuard {
 public:
  MluDeviceGuard(int device_id) {
    CHECK_CNRT(cnrtSetDevice, device_id);
  }
};  // class MluDeviceGuard

class Record
{
public:
  Record(std::string filename)
  {
    outfile.open(("output/" + filename).c_str(), std::ios::trunc | std::ios::out);
  }

  ~Record()
  {
    if (outfile.is_open())
      outfile.close();
  }

  void write(std::string line, bool print = false)
  {
    outfile << line << std::endl;
    if (print)
    {
      std::cout << line << std::endl;
    }
  }

private:
  std::ofstream outfile;
};

inline static void PrintModelInfo(magicmind::IModel *model) {
  LOG(INFO) << "==================Model info===================";
  LOG(INFO) << "Input number : " << model->GetInputNum();
  for (int i = 0; i < model->GetInputNum(); ++i)
    LOG(INFO) << "input[" << i << "] : dimensions "
              << model->GetInputDimension(i) << ", data type ["
              << model->GetInputDataType(i) << "]";
  LOG(INFO) << "Output number : " << model->GetOutputNum();
  for (int i = 0; i < model->GetOutputNum(); ++i)
    LOG(INFO) << "output[" << i << "] : dimensions "
              << model->GetOutputDimension(i) << ", data type ["
              << model->GetOutputDataType(i) << "]";
}

inline std::vector<std::string> split(const std::string &in, const std::string &delim)
{
    std::regex re{delim};
    return std::vector<std::string>{
        std::sregex_token_iterator(in.begin(), in.end(), re, -1), std::sregex_token_iterator()};
}

inline bool check_file_exist(std::string path)
{
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0)
  {
    if ((buffer.st_mode & S_IFDIR) == 0)
    {
      return true;
    }
    return false;
  }
  return false;
}

inline bool check_folder_exist(std::string path)
{
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0)
  {
    if ((buffer.st_mode & S_IFDIR) == 0)
    {
      return false;
    }
    return true;
  }
  return false;
}

// The implementation of this function shows the limitations of the current
// program on the model.
static bool CheckModel(magicmind::IModel *model) {
  if (model->GetInputNum() != 1) {
    LOG(ERROR) << "Input number is [" << model->GetInputNum() << "].";
    return false;
  }
  if (model->GetOutputNum() != 1) {
    LOG(ERROR) << "Output number is [" << model->GetOutputNum() << "].";
    return false;
  }
  if (model->GetInputDimension(0)[3] != 3) {
    LOG(ERROR) << "Input channels [" << model->GetInputDimension(0)[3] << "].";
    return false;
  }
  if (model->GetInputDataType(0) != magicmind::DataType::UINT8) {
    LOG(ERROR) << "Input data type is [" << model->GetInputDataType(0) << "].";
    return false;
  }
  if (model->GetOutputDataType(0) != magicmind::DataType::FLOAT32) {
    LOG(ERROR) << "Output data type is [" << model->GetOutputDataType(0)
               << "].";
    return false;
  }
  return true;
}

inline std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  LOG_IF(FATAL, std::string::npos == slash_pos)
    << "[" << abs_path << "] is not an absolute path.";
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  LOG_IF(FATAL, point_pos == slash_pos + 1)
    << "[" << abs_path << "] is not a file path.";
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

inline std::vector<std::string> LoadFileList(const std::string &path) {
  std::ifstream ifs(path);
  LOG_IF(FATAL, !ifs.is_open()) << "Open label file failed. path : " << path;
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(ifs, line)) {
    labels.emplace_back(std::move(line));
  }
  ifs.close();
  return labels;
}

// Split string with space
inline static std::vector<std::string> SplitString(const std::string line) {
  std::vector<std::string> res;
  std::string result;
  // read line to input
  std::stringstream input(line);
  // output to result and save in res
  while (input >> result) res.push_back(result);
  return res;
}

struct KeypointInfo {
  cv::Point2f pos;  // position
  float score;
};  // struct KeypointInfo

// std::vector<std::vector<>>: first dimension for body parts
// second dimension for detected keypoints on each body parts
using Keypoints = std::vector<std::vector<KeypointInfo>>;

static cv::Point kInvalidPos(-1, -1);
struct PersonInfo {
  /**
   * keypoint index in Keypoints(type std::vector<std::vector<>>)
   * cv::Point(-1, -1) means invalid keypoint
   * size = NetParams::nbody_parts
   **/
  std::vector<cv::Point> keypoint_idxs;
  int nkeypoints;
  float score;
};  // struct PersonInfo

using PersonInfos = std::vector<PersonInfo>;

inline
cv::Point RoundPoint(const cv::Point2f &point, const cv::Mat &img) {
  return cv::Point(std::min(static_cast<int>(std::round(point.x)), img.cols - 1),
                   std::min(static_cast<int>(std::round(point.y)), img.rows - 1));
}

#endif // UTILS_HPP
