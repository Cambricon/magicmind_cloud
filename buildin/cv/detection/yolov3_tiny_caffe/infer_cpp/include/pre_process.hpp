#ifndef _C3D_PRE_PROCESS_HPP
#define _C3D_PRE_PROCESS_HPP

#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include <deque>
using namespace std;

std::vector<cv::String> LoadImages(int batch_size,const string& dataset_path);
float Preprocess(cv::Mat img, const int h,const int w, cv::Mat& dst,uint8_t pad_value);

#endif //_C3D_PRE_PROCESS_HPP