#include "../include/pre_process.hpp"
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
result *LoadImages(const std::string image_dir, int batch_size, const std::string file_list) {
  result *test = new result;
  char abs_path[PATH_MAX];
  if (realpath(image_dir.c_str(), abs_path) == NULL) {
    std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
    exit(1);
  }
  std::string glob_path = std::string(abs_path);
  std::ifstream in(file_list);
  std::string line;
  std::string image_name;
  int count = 0;
  std::string image_path;
  int label;
  while (getline(in, line)) {
    int found = line.find(" ");
    image_name = line.substr(0, found);
    label = std::stoi(line.substr(found + 1));
    image_path = glob_path + "/" + image_name;
    test->image_paths.push_back(image_path);
    test->labels.push_back(label);
  }
  return test;
}

DEFINE_int32(new_size, 256, "The resized image size.");
DEFINE_bool(input_rgb, false, "Input convert to rgb or not.");

// resize to new_size + center crop to input_dim
cv::Mat Preprocess(cv::Mat img, const int h, const int w) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
      (x - mean) / std : This calculation process is performed at the first layer of the model,
      See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
  */

  if (h > FLAGS_new_size) {
    std::cout << "new_size[" << FLAGS_new_size << "] less than h [" << h << "]." << std::endl;
  }
  if (w > FLAGS_new_size) {
    std::cout << "new_size[" << FLAGS_new_size << "] less than w [" << w << "]." << std::endl;
  }
  // resize
  cv::Mat resized;
  float scale = 1.0f * FLAGS_new_size / std::min(img.cols, img.rows);
  cv::resize(img, resized, cv::Size(std::round(scale * img.cols), std::round(scale * img.rows)));
  // center crop
  auto roi = resized(cv::Rect((resized.cols - w) / 2, (resized.rows - h) / 2, w, h));
  cv::Mat dst_img(h, w, CV_8UC3, cv::Scalar(0, 0, 255));
  if (FLAGS_input_rgb) {
    cv::cvtColor(roi, dst_img, cv::COLOR_BGR2RGB);
  } else {
    roi.copyTo(dst_img);
  }
  return dst_img;
}
