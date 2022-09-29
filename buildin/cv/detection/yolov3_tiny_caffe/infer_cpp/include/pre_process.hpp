#ifndef PRE_PROCESS_HPP
#define PRE_PROCESS_HPP

#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include <deque>
using namespace std;

std::vector<cv::String> LoadImages(int batch_size, const string& dataset_path);
float Preprocess(cv::Mat img, const int h, const int w, cv::Mat& dst,
                 uint8_t pad_value);

#endif  // PRE_PROCESS_HPP