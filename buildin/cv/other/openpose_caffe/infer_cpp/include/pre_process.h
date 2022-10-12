#ifndef _PRE_PROCESS_H
#define _PRE_PROCESS_H

#include <cnrt.h>
#include <mm_runtime.h>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <regex>

float Preprocess(cv::Mat img, const magicmind::Dims &input_dim, uint8_t *dst);

std::vector<std::string> LoadImages(const std::string image_dir,
                                    const std::string image_list,
                                    const int batch_size);

#endif  //_PRE_PROCESS_H
