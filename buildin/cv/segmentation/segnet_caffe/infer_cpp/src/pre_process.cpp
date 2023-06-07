#include <fstream>
#include <iostream>
#include "pre_process.h"
//#include "utils.h"

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir,
                                    const std::string image_list,
                                    const int batch_size) {
  char abs_path[PATH_MAX];
  std::ifstream cur_ifs(image_list);
  if (!cur_ifs.is_open()) {
    std::cout << "Image name list file [" << image_list << "] not exists." << std::endl;
  }
  std::string line;
  std::vector<std::string> image_paths;
  while (getline(cur_ifs, line))
    image_paths.emplace_back(image_dir + "/" + line + ".jpg");
  cur_ifs.close();
  return image_paths;
}
cv::Mat Preprocess(cv::Mat img, int dst_h, int dst_w) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  cv::Mat resized(dst_h, dst_w, CV_8UC3);
  cv::resize(img, resized, cv::Size(dst_w, dst_h));
  return resized;
}
