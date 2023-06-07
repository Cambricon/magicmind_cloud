#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// bounding box
struct BBox {
  float score;
  int cx, cy, w, h;
  std::vector<cv::Point2f> landms;
};  // struct BBox

std::vector<BBox> Postprocess(
    cv::Mat img, std::vector<int64_t> &input_dim,
    std::vector<const float*> outputs,
    std::vector<std::vector<int64_t>> output_dims);

// Gets file name without extension from the absolute path of the file
std::string GetFileName(const std::string &abs_path);

#endif

