#ifndef _POST_PROCESS_HPP
#define _POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "../include/utils.hpp"

std::vector<BBox> Postprocess(
    cv::Mat img, const magicmind::Dims &input_dim,
    std::vector<const float*> outputs,
    std::vector<magicmind::Dims> output_dims);

#endif

