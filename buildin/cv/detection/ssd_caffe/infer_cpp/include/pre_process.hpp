#ifndef _SAMPLE_PRE_PROCESS_HPP
#define _SAMPLE_PRE_PROCESS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


std::vector<std::string> LoadImages(const std::string image_dir, const int batch_size);
cv::Mat process_img(cv::Mat src_img, bool transpose = false, bool normlize = true, bool swapBR = false, int depth = CV_8U);

#endif //_SAMPLE_PRE_PROCESS_HPP

