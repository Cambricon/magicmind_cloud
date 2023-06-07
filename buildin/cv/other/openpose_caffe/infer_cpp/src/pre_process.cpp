#include "pre_process.h"
#include "utils.h"
//#include <glog/logging.h>
#include <macros.h>


/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir, const std::string image_list) {
  std::vector<std::string> image_paths = LoadFileList(image_list);
  return image_paths;
}

float Preprocess(cv::Mat img, const magicmind::Dims &input_dim, uint8_t *dst) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
     (x - mean) / std : This calculation process is performed at the first layer of the model,
     See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
  */
  // resize as latter box
  int dst_h = input_dim[1];
  int dst_w = input_dim[2];
  float scaling_factor = std::min(1.0f * (dst_h - 1) / (img.rows - 1), 1.0f * (dst_w - 1) / (img.cols - 1));
  cv::Mat dst_mat(dst_h, dst_w, CV_8UC3, dst);
  cv::Mat m = cv::Mat::eye(2, 3, CV_64F);
  m.at<double>(0, 0) = scaling_factor;
  m.at<double>(1, 1) = scaling_factor;
  cv::warpAffine(img, dst_mat, m, cv::Size(dst_w, dst_h),
    (scaling_factor > 1.f ? cv::INTER_CUBIC : cv::INTER_AREA), cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  return scaling_factor;
}

