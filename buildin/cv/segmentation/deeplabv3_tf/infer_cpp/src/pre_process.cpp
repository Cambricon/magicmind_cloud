#include "../include/pre_process.h"
#include "../include/utils.h"

std::vector<cv::Mat> LoadCocoImages(std::string coco_path, int count) {
  std::vector<cv::Mat> imgs;
  std::string image_path = coco_path + "/images";
  if (!check_folder_exist(image_path)) {
    std::cout << "image folder: " + image_path + " does not exist.\n";
    exit(0);
  }
  std::vector<cv::String> image_files;
  cv::glob(image_path + "/*.jpg", image_files);
  int current_count = 0;
  for (int i = 0; i < count; ++i) {
    cv::Mat img = cv::imread(image_files[i]);
    if (img.empty()) {
      std::cout << "failed to load image " + image_files[i] + ".\n";
      exit(0);
    }
    imgs.push_back(img);
  }
  if (imgs.empty()) {
    std::cout << " image folder: " + image_path + "not contian jpg file.\n";
    exit(0);
  }
  return imgs;
}

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
std::vector<cv::String> LoadImages(const std::string image_dir,
                                   const int batch_size, int image_num,
                                   const std::string file_list) {
  char abs_path[PATH_MAX];
  if (!realpath(image_dir.c_str(), abs_path)) {
    std::cout << "Get real image path failed." << std::endl;
  }
  std::string glob_path = std::string(abs_path);
  std::ifstream in;
  std::string image_name;
  std::string image_path;
  std::vector<cv::String> image_paths;
  in.open(file_list);
  char c;
  int lineCnt=0;
  while(in.get(c)) {
  	if(c=='\n')
  	lineCnt++;
  }
  if (image_num > lineCnt) {
    std::cout << "image_num now is " << image_num << ", but it must less than images of " << file_list << ": " << lineCnt << std::endl;
    abort();
  }
  in.close();
  in.seekg(0);
  in.open(file_list);
  int count = 0;
  while (getline(in, image_name)) {
    image_path = glob_path + '/' + image_name + ".jpg";
    image_paths.push_back(image_path);
    count += 1;
    if (count >= image_num) {
      break;
    }
  }
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

cv::Mat Preprocess(cv::Mat src_img) {
  int src_h = src_img.rows;
  int src_w = src_img.cols;
  int dst_h = 513;
  int dst_w = 513;
  // resize
  cv::resize(src_img, src_img, cv::Size(dst_h, dst_w), cv::INTER_AREA);
  // bgr to rgb
  cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
  return src_img;
}
