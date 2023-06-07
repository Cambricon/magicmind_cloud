#include "post_process.hpp"
#include "utils.hpp"

std::map<int, std::string> load_name(std::string name_map_file) {
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
  float scale_w = ratio * src_w;
  float scale_h = ratio * src_h;
  int detect_num = results.size();
  for (int i = 0; i < detect_num; ++i) {
    int detect_class = results[i][0];
    float score = results[i][1];
    float xmin = results[i][2];
    float ymin = results[i][3];
    float xmax = results[i][4];
    float ymax = results[i][5];
    xmin = std::max(float(0.0), std::min(xmin, dst_w));
    xmax = std::max(float(0.0), std::min(xmax, dst_w));
    ymin = std::max(float(0.0), std::min(ymin, dst_h));
    ymax = std::max(float(0.0), std::min(ymax, dst_h));
    xmin = (xmin - (dst_w - scale_w) / 2) / ratio;
    ymin = (ymin - (dst_h - scale_h) / 2) / ratio;
    xmax = (xmax - (dst_w - scale_w) / 2) / ratio;
    ymax = (ymax - (dst_h - scale_h) / 2) / ratio;
    xmin = std::max(0.0f, float(xmin));
    xmax = std::max(0.0f, float(xmax));
    ymin = std::max(0.0f, float(ymin));
    ymax = std::max(0.0f, float(ymax));
    file_map << name_map[detect_class] << "," << score << "," << xmin << "," << ymin << "," << xmax
             << "," << ymax << "\n";
    if (save_img) {
      cv::rectangle(img, cv::Rect(cv::Point(int(xmin), int(ymin)), cv::Point(int(xmax), int(ymax))),
                    cv::Scalar(0, 255, 0));
      auto fontface = cv::FONT_HERSHEY_TRIPLEX;
      double fontscale = 0.5;
      int thickness = 1;
      int baseline = 0;
      std::string text = name_map[detect_class] + ": " + std::to_string(score);
      cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
      cv::putText(img, text, cv::Point(int(xmin), int(ymin) + text_size.height), fontface,
                  fontscale, cv::Scalar(255, 255, 255), thickness);
    }
  }
  if (save_img) {
    imwrite(output_dir + "/" + name + ".jpg", img);
  }
  file_map.close();
  return true;
}
