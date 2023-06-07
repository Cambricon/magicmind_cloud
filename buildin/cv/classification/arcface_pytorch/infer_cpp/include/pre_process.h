#ifndef _PRE_PROCESS_H
#define _PRE_PROCESS_H

#include <map>
#include <regex>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "logger.h"

cv::Mat Preprocess(cv::Mat img,
                   std::vector<int64_t> &input_dim,
                   const std::vector<std::string> landmarks);

std::vector<std::string> LoadImages(const std::string image_dir,
                                    const std::string image_list,
                                    int image_num,
                                    const int batch_size);

inline std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  if (std::string::npos == slash_pos)
    SLOG(INFO) << "[" << abs_path << "] is not an absolute path.";
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  if (point_pos == slash_pos + 1)
    SLOG(INFO) << "[" << abs_path << "] is not a file path.";
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

inline std::vector<std::string> LoadFileList(const std::string &path, int num) {
  std::ifstream ifs(path);
  if (!ifs.is_open())
    SLOG(INFO) << "Open label file failed. path : " << path;
  std::vector<std::string> labels;
  std::string line;
  int count = 0;
  while (std::getline(ifs, line)) {
    labels.emplace_back(std::move(line));
    ++count;
    if (count == num)
      break;
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
  while (input >> result)
    res.push_back(result);
  return res;
}

#endif  //_PRE_PROCESS_H
