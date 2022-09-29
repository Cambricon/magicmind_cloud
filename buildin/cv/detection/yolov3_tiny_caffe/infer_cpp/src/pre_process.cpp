#include "pre_process.hpp"
#include "utils.hpp"

std::vector<cv::String> LoadImages(int batch_size, const string& dataset_path) {
  char abs_path[PATH_MAX];
  CHECK_NE(true, dataset_path.empty());
  if (!realpath(dataset_path.c_str(), abs_path)) {
    cout << "Get real image path failed.";
  }
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> image_paths;
  cv::glob(glob_path + "/*.jpg", image_paths, false);
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size) {
    cout << "There are " << image_paths.size() << " images in total, add "
         << pad_num
         << " more images to make the number of images is an integral multiple "
            "of batchsize["
         << batch_size << "]." << endl;
    while (pad_num--) image_paths.emplace_back(*image_paths.rbegin());
  }
  return image_paths;
}

/**
  @return Returns resized image and scaling factors
 */
std::pair<cv::Mat, float> LatterBox(cv::Mat img, int dst_h, int dst_w,
                                    uint8_t pad_value) {
  float scaling_factors =
      std::min(1.0f * dst_h / img.rows, 1.0f * dst_w / img.cols);
  int unpad_h = std::floor(scaling_factors * img.rows);
  int unpad_w = std::floor(scaling_factors * img.cols);
  int pad_h = dst_h - unpad_h;
  int pad_w = dst_w - unpad_w;
  int pad_top = std::floor(pad_h / 2.0f);
  int pad_left = std::floor(pad_w / 2.0f);
  int pad_bottom = pad_h - pad_top;
  int pad_right = pad_w - pad_left;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(unpad_w, unpad_h));
  cv::Mat dst;
  cv::copyMakeBorder(resized, dst, pad_top, pad_bottom, pad_left, pad_right,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(pad_value, pad_value, pad_value));
  return std::make_pair(dst, scaling_factors);
}

/**
 * @return Returns the scaling factors in resize
 * scaling factors is useful for get detection results.
 */
float Preprocess(cv::Mat img, const int h, const int w, cv::Mat& dst,
                 uint8_t pad_value) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
      (x - mean) / std : This calculation process is performed at the first
     layer of the model, See parameter named [insert_bn_before_firstnode] in
     magicmind::IBuildConfig.
  */
  // resize as latter box
  auto ret = LatterBox(img, h, w, pad_value);
  cv::cvtColor(ret.first, dst, cv::COLOR_BGR2RGB);
  return ret.second;
}