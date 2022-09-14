#ifndef _SAMPLE_POST_PROCESS_HPP
#define _SAMPLE_POST_PROCESS_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../include/utils.hpp"

std::map<int, std::string> load_name(std::string name_map_file);

std::vector<BBox> Postprocess(const std::vector<float *> &outputs, const std::vector<magicmind::Dims> &output_dims, int max_num, float thresholds);

void RescaleBBox(cv::Mat img, const magicmind::Dims &heatmap_dims, std::vector<BBox> &bboxes, std::map<int, std::string> imagenet_name_map, const std::string name, const std::string output_dir);

void Draw(cv::Mat img, const std::vector<BBox> &bboxes, std::map<int, std::string> imagenet_name_map);
#endif

