#ifndef _PRE_PROCESS_HPP
#define _PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <gflags/gflags.h>

struct result{
    std::vector<std::string> image_paths;
    std::vector<int> labels;
 };
result * LoadImages(const std::string image_dir, const int batch_size, int image_num, const std::string file_list);

cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim, int input_size, bool input_rgb);
#endif //_PRE_PROCESS_HPP