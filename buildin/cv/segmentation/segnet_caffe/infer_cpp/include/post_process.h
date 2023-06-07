#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mm_runtime.h>

cv::Mat PostProcess(cv::Mat &img, const magicmind::Dims &preds_dim, float *preds);

#endif  //_POST_PROCESS_HPP
