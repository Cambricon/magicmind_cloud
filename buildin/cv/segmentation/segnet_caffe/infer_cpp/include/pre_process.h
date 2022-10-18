#ifndef _PRE_PROCESS_HPP
#define _PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat Preprocess(cv::Mat src_img,int dst_h, int dst_w);

std::vector<std::string> LoadImages(const std::string image_dir, const std::string image_list, const int batch_size);

#endif //_PRE_PROCESS_HPP
