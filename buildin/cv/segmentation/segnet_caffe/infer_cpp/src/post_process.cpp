#include "post_process.h"
#include <mm_runtime.h>
//#include "utils.h"
static inline int ArgMax(float *begin, float *end) {
  return std::distance(begin, std::max_element(begin, end));
}

cv::Mat PostProcess(cv::Mat &img, const magicmind::Dims &preds_dim, float *preds) {
  int output_h = preds_dim[1];
  int output_w = preds_dim[2];
  int nclasses = preds_dim[3];
  cv::Mat ret = cv::Mat::zeros(output_h, output_w, CV_8UC1);
  for (int y = 0; y < output_h; ++y) {
    int y_offset = y * output_w * nclasses;
    for (int x = 0; x < output_w; ++x) {
      float *tpred = preds + y_offset + x * nclasses;
      ret.at<uint8_t>(y, x) = static_cast<uint8_t>(ArgMax(tpred, tpred + nclasses));
    }
  }
  cv::Mat resized;
  cv::resize(ret, resized, img.size(), cv::INTER_NEAREST);
  return resized;
}
