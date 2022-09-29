#ifndef POST_PROCESS_HPP
#define POST_PROCESS_HPP

#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include <deque>
using namespace std;
static const std::unordered_map<std::string, int> kCOCOPaperCategoryIdMap = {
    {"person", 1},          {"bicycle", 2},        {"car", 3},
    {"motorbike", 4},       {"aeroplane", 5},      {"bus", 6},
    {"train", 7},           {"truck", 8},          {"boat", 9},
    {"traffic light", 10},  {"fire hydrant", 11},  {"street sign", 12},
    {"stop sign", 13},      {"parking meter", 14}, {"bench", 15},
    {"bird", 16},           {"cat", 17},           {"dog", 18},
    {"horse", 19},          {"sheep", 20},         {"cow", 21},
    {"elephant", 22},       {"bear", 23},          {"zebra", 24},
    {"giraffe", 25},        {"hat", 26},           {"backpack", 27},
    {"umbrella", 28},       {"shoe", 29},          {"eye glasses", 30},
    {"handbag", 31},        {"tie", 32},           {"suitcase", 33},
    {"frisbee", 34},        {"skis", 35},          {"snowboard", 36},
    {"sports ball", 37},    {"kite", 38},          {"baseball bat", 39},
    {"baseball glove", 40}, {"skateboard", 41},    {"surfboard", 42},
    {"tennis racket", 43},  {"bottle", 44},        {"plate", 45},
    {"wine glass", 46},     {"cup", 47},           {"fork", 48},
    {"knife", 49},          {"spoon", 50},         {"bowl", 51},
    {"banana", 52},         {"apple", 53},         {"sandwich", 54},
    {"orange", 55},         {"broccoli", 56},      {"carrot", 57},
    {"hot dog", 58},        {"pizza", 59},         {"donut", 60},
    {"cake", 61},           {"chair", 62},         {"sofa", 63},
    {"potted plant", 64},   {"bed", 65},           {"mirror", 66},
    {"diningtable", 67},    {"window", 68},        {"desk", 69},
    {"toilet", 70},         {"door", 71},          {"tvmonitor", 72},
    {"laptop", 73},         {"mouse", 74},         {"remote", 75},
    {"keyboard", 76},       {"cell phone", 77},    {"microwave", 78},
    {"oven", 79},           {"toaster", 80},       {"sink", 81},
    {"refrigerator", 82},   {"blender", 83},       {"book", 84},
    {"clock", 85},          {"vase", 86},          {"scissors", 87},
    {"teddy bear", 88},     {"hair drier", 89},    {"toothbrush", 90},
    {"hair brus", 91}};

// bounding box
struct BBox {
  int label;
  float score;
  int left;
  int top;
  int right;
  int bottom;
};  // struct BBox

// draw bboxes on image
void Draw(cv::Mat img, const std::vector<BBox>& bboxes,
          const std::vector<std::string>& labels);
std::vector<BBox> Yolov3GetBBox(cv::Mat img, float scaling_factors,
                                const int input_h, const int input_w,
                                float* bbox_output, int bbox_num,
                                float confidence_thresholds);

void saveImgAndPreds(bool save_img_, bool save_pred_,
                     vector<cv::Mat>& src_imgs_, vector<string>& image_name_,
                     const string& save_img_dir_, const string& save_pred_dir_,
                     const std::vector<std::string>& labels_, const int num_,
                     const vector<BBox>& bboxes_);

#endif  // POST_PROCESS_HPP