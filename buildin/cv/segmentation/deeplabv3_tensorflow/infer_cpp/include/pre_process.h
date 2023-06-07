#ifndef _PRE_PROCESS_HPP
#define _PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

std::vector<cv::Mat> LoadCocoImages(std::string coco_path, int count = 10);

cv::Mat Preprocess(cv::Mat src_img);

std::vector<cv::String> LoadImages(const std::string image_dir,
                                   const int batch_size, int image_num,
                                   const std::string file_list);

#endif  //_PRE_PROCESS_HPP
