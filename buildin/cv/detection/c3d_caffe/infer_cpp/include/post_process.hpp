#ifndef POST_PROCESS_HPP
#define POST_PROCESS_HPP

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
std::vector<int> ArgTopK(const float *data, int classes, int k);
#endif  // POST_PROCESS_HPP