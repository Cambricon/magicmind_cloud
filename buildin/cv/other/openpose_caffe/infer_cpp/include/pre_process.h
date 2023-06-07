#ifndef _PRE_PROCESS_H
#define _PRE_PROCESS_H

#include <map>
#include <regex>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

float Preprocess(cv::Mat img, const magicmind::Dims &input_dim, uint8_t *dst);

std::vector<std::string> LoadImages(const std::string image_dir, const std::string image_list);

#endif //_PRE_PROCESS_H

