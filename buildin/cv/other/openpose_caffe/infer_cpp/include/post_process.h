#ifndef _POST_PROCESS_H
#define _POST_PROCESS_H

#include <utility>
#include <mm_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "utils.h"
std::pair<Keypoints, PersonInfos> Postprocess(
      cv::Mat img, float *preds, const float scaling_factor,
      const magicmind::Dims &input_dim, const magicmind::Dims &output_dim);
#endif //_POST_PROCESS_H
