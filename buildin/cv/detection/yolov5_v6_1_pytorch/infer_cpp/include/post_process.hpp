#ifndef _SAMPLE_POST_PROCESS_HPP
#define _SAMPLE_POST_PROCESS_HPP

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

std::map<int, std::string> load_name(std::string name_map_file);

bool post_process(cv::Mat &img, std::vector<std::vector<float>> results,
                  std::map<int, std::string> coco_name_map,
                  const std::string image_name, const std::string output_dir,
                  bool save_img);

#endif
