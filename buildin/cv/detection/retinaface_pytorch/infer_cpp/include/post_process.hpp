#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "../include/utils.hpp"

std::vector<BBox> Postprocess(cv::Mat img, const magicmind::Dims &input_dim,
                              std::vector<const float *> outputs,
                              std::vector<magicmind::Dims> output_dims);

#endif
