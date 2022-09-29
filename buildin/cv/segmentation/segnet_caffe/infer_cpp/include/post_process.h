#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <gflags/gflags.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "utils.h"

cv::Mat PostProcess(cv::Mat &img, const magicmind::Dims &preds_dim,
                    float *preds);

#endif  //_POST_PROCESS_HPP
