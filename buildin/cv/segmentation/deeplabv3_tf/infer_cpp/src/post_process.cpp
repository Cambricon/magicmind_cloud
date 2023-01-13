#include "../include/post_process.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../include/utils.h"

typedef cv::Vec<uchar, 3> Vec3b;
std::map<int, std::string> LoadLabelName(std::string name_map_file) {
  if (!check_file_exist(name_map_file)) {
    std::cout << "coco_name file: " + name_map_file + " does not exist.\n";
    exit(0);
  }
  std::map<int, std::string> coco_name_map;
  std::ifstream in(name_map_file);
  if (!in) {
    std::cout << "failed to load coco_name file: " + name_map_file + ".\n";
    exit(0);
  }
  std::string line;
  int index = 0;
  while (getline(in, line)) {
    coco_name_map[index] = line;
    index += 1;
  }
  return coco_name_map;
}

void PostProcess(uint32_t *ptr, uint32_t output_h, uint32_t output_w,
                 const std::string name, const std::string output_dir,
                 bool save_img) {
  uint32_t rows = output_h;
  uint32_t cols = output_w;
  char *out_ptr = new char[rows * cols * 3];
  cv::Mat color_out(rows, cols, CV_8UC3, out_ptr);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      int label = static_cast<int>(((uint32_t *)ptr)[i * cols + j]);
      // save RGB
      out_ptr[i * cols * 3 + j * 3] = colormap[label][2];
      out_ptr[i * cols * 3 + j * 3 + 1] = colormap[label][1];
      out_ptr[i * cols * 3 + j * 3 + 2] = colormap[label][0];
    }
  }
  if (save_img) {
    imwrite(output_dir + "/" + name + ".png", color_out);
  }
  delete out_ptr;
}
