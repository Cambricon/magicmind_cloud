#ifndef _PRE_PROCESS_H
#define _PRE_PROCESS_H

#include <map>
#include <regex>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim, const std::vector<std::string> landmarks);

std::vector<std::string> LoadImages(const std::string image_dir, const std::string image_list, const int batch_size);

#endif //_PRE_PROCESS_H

