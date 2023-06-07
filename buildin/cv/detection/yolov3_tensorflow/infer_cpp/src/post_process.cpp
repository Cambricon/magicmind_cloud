#include "post_process.hpp"
#include "utils.hpp"
#include "macros.h"

std::map<int, std::string> load_name(std::string name_map_file) {
  if (!check_file_exist(name_map_file)) {
    SLOG(ERROR) << "coco_name file: " + name_map_file + " does not exist.\n";
    exit(0);
  }
  std::map<int, std::string> coco_name_map;
  std::ifstream in(name_map_file);
  if (!in) {
    SLOG(ERROR) << "failed to load coco_name file: " + name_map_file + ".\n";
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

bool post_process(cv::Mat &img,
                  std::vector<std::vector<float>> results,
                  std::map<int, std::string> name_map,
                  const std::string name,
                  const std::string output_dir,
                  bool save_img,
                  float dst_h,
                  float dst_w) {
  std::string filename = output_dir + "/" + name + ".txt";
  std::ofstream file_map(filename);
  int src_h = img.rows;
  int src_w = img.cols;
  float ratio = std::min(float(dst_h) / float(src_h), float(dst_w) / float(src_w));
  float scale_w = std::floor(src_w * ratio);
  float scale_h = std::floor(src_h * ratio);
  float pad_top = (dst_h - ratio * img.rows) / 2;
  float pad_left = (dst_w - ratio * img.cols) / 2;
  int detect_num = results.size();
  for (int i = 0; i < detect_num; ++i) {
    int detect_class = results[i][0];
    float score = results[i][1];
    float xmin = (results[i][2] * dst_w - pad_left) / ratio;
    float ymin = (results[i][3] * dst_h - pad_top) / ratio;
    float xmax = (results[i][4] * dst_w - pad_left) / ratio;
    float ymax = (results[i][5] * dst_h - pad_top) / ratio;
    xmin = std::max(float(0.0), std::min(xmin, float(src_w)));
    xmax = std::max(float(0.0), std::min(xmax, float(src_w)));
    ymin = std::max(float(0.0), std::min(ymin, float(src_h)));
    ymax = std::max(float(0.0), std::min(ymax, float(src_h)));
    file_map << name_map[detect_class] << "," << score << "," << xmin << "," << ymin << "," << xmax
             << "," << ymax << "\n";
    if (save_img) {
      cv::rectangle(img, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)),
                    cv::Scalar(0, 255, 0));
      auto fontface = cv::FONT_HERSHEY_TRIPLEX;
      double fontscale = 0.5;
      int thickness = 1;
      int baseline = 0;
      std::string text = name_map[detect_class] + ": " + std::to_string(score);
      cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
      cv::putText(img, text, cv::Point(xmin, ymin + text_size.height), fontface, fontscale,
                  cv::Scalar(255, 255, 255), thickness);
    }
  }
  if (save_img) {
    imwrite(output_dir + "/" + name + ".jpg", img);
  }
  file_map.close();
  return true;
}
