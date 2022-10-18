#ifndef NET_PARAMS_H_
#define NET_PARAMS_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct NetParams {
  int nbody_parts;
  int nbody_part_pairs;
  int nheatmaps;
  std::vector<cv::Point> body_part_pairs;
  std::vector<cv::Point> paf_indexs;

  // render params
  std::vector<cv::Scalar> colors;
  std::vector<cv::Point> render_body_part_pairs;

  // coco
  std::vector<int> indexes_in_coco_order;
};  // struct NetParams

// get network parameters by flag[network], BODY_25 and COCO are supported
const NetParams &GetNetParams();

const std::string &GetNetName();

#endif  // NET_PARAMS_H_

