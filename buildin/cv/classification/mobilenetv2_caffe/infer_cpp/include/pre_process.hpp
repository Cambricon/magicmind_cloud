#ifndef PRE_PROCESS_HPP
#define PRE_PROCESS_HPP
#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <deque>
using namespace std;

struct result{
    std::vector<std::string> image_paths;
    std::vector<int> labels;
 };
result * LoadImages(const std::string image_dir, const int batch_size, const std::string file_list);
void Preprocess(cv::Mat img, const int h,const int w, cv::Mat& dst);

#endif //PRE_PROCESS_HPP