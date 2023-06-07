#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>
#include <mm_runtime.h>
//#include <glog/logging.h>
#include <cnrt.h>
#include <vector>
#include <regex>
#include "sys/stat.h"
#include "net_params.h"
#include "macros.h"

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
    SLOG(ERROR) << "Input number is [" << model->GetInputNum() << "].";
    return false;
  }
  if (model->GetOutputNum() != 1) {
    SLOG(ERROR) << "Output number is [" << model->GetOutputNum() << "].";
    return false;
  }
  if (model->GetInputDimension(0)[3] != 3) {
    SLOG(ERROR) << "Input channels [" << model->GetInputDimension(0)[3] << "].";
    return false;
  }
  if (model->GetInputDataType(0) != magicmind::DataType::UINT8) {
    SLOG(ERROR) << "Input data type is [" << model->GetInputDataType(0) << "].";
    return false;
  }
  if (model->GetOutputDataType(0) != magicmind::DataType::FLOAT32) {
    SLOG(ERROR) << "Output data type is [" << model->GetOutputDataType(0)
               << "].";
    return false;
  }
  return true;
}

inline std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  if (std::string::npos == slash_pos){
      SLOG(ERROR) << "[" << abs_path << "] is not an absolute path.";
  }
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  if (point_pos == slash_pos + 1){
      SLOG(ERROR) << "[" << abs_path << "] is not a file path.";
  }
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

inline std::vector<std::string> LoadFileList(const std::string &path) {
  std::ifstream ifs(path);
  if(!ifs.is_open()){
  	SLOG(ERROR) << "Open label file failed. path : " << path;
  }
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
