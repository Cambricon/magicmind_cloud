#include "pre_process.h"
#include "utils.h"

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir,
                                    const std::string image_list,
                                    const int batch_size) {
  char abs_path[PATH_MAX];
  std::ifstream ifs(image_list);
  if (!ifs.is_open()) {
    std::cout << "Image name list file [" << image_list << "] not exists."
              << std::endl;
  }
  std::string line;
  std::vector<std::string> image_paths;
  while (getline(ifs, line))
    image_paths.emplace_back(image_dir + "/" + line + ".jpg");
  ifs.close();
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer
  // multiple of the batch size
  size_t pad_num = batch_size - image_paths.size() % batch_size;
  if (pad_num != batch_size) {
    std::cout << "There are " << image_paths.size() << " images in total, add "
              << pad_num
              << " more images to make the number of images is an integral "
                 "multiple of batchsize["
              << batch_size << "]." << std::endl;
    while (pad_num--) {
      image_paths.emplace_back(*image_paths.rbegin());
    }
  }
  return image_paths;
}

cv::Mat Preprocess(cv::Mat img, int dst_h, int dst_w) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  cv::Mat resized(dst_h, dst_w, CV_8UC3);
  cv::resize(img, resized, cv::Size(dst_w, dst_h));
  return resized;
}
