#include "pre_process.hpp"
#include "utils.hpp"
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
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch
  // size size_t pad_num = batch_size - test->image_paths.size() % batch_size; if (pad_num !=
  // batch_size) {
  //     std::cout << "There are " << test->image_paths.size() << " images in total, add " <<
  //     pad_num
  //         << " more images to make the number of images is an integral multiple of batchsize[" <<
  //         batch_size << "].";
  //     while (pad_num--)
  //     test->image_paths.emplace_back(*test->image_paths.rbegin());
  // }
  return test;
}

// resize to new_size + center crop to input_dim
void Preprocess(cv::Mat img, const int h, const int w, cv::Mat &dst) {
  // NHWC order implementation. Make sure your model's input is in NHWC order.
  /*
      (x - mean) / std : This calculation process is performed at the first layer of the model,
      See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
  */
  // resize
  int new_size = 256;
  float scale = 1.0f * new_size / std::min(img.cols, img.rows);
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(std::round(scale * img.cols), std::round(scale * img.rows)));
  // center crop
  auto roi = resized(cv::Rect((resized.cols - w) / 2, (resized.rows - h) / 2, w, h));
  roi.copyTo(dst);
}