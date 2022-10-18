#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "net_params.h"
#include "utils.h"
#include "coco_result_saver.h"

static const std::map<unsigned int, std::string> kBody25PartStrs {
  {0,  "Nose"},
  {1,  "Neck"},
  {2,  "RShoulder"},
  {3,  "RElbow"},
  {4,  "RWrist"},
  {5,  "LShoulder"},
  {6,  "LElbow"},
  {7,  "LWrist"},
  {8,  "MidHip"},
  {9,  "RHip"},
  {10, "RKnee"},
  {11, "RAnkle"},
  {12, "LHip"},
  {13, "LKnee"},
  {14, "LAnkle"},
  {15, "REye"},
  {16, "LEye"},
  {17, "REar"},
  {18, "LEar"},
  {19, "LBigToe"},
  {20, "LSmallToe"},
  {21, "LHeel"},
  {22, "RBigToe"},
  {23, "RSmallToe"},
  {24, "RHeel"},
  {25, "Background"}
};

static const std::map<unsigned int, std::string> kBodyCOCOPartStrs {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "RHip"},
    {9,  "RKnee"},
    {10, "RAnkle"},
    {11, "LHip"},
    {12, "LKnee"},
    {13, "LAnkle"},
    {14, "REye"},
    {15, "LEye"},
    {16, "REar"},
    {17, "LEar"},
    {18, "Background"}
};

COCOResultSaver::COCOResultSaver(const std::string &output_file)
    : writer_(stream_) {
  if (!output_file.empty()) {
    stream_.ofs.open(output_file);
    CHECK_EQ(true, stream_.ofs.is_open());
    writer_.StartArray();
  }
}

COCOResultSaver::~COCOResultSaver() {
  if (stream_.ofs.is_open()) {
    writer_.EndArray();
    writer_.Flush();
    stream_.ofs.close();
  }
}

void COCOResultSaver::Write(
    cv::Mat img,
    const std::string &image_name,
    const Keypoints &keypoints,
    const PersonInfos &persons) {
  // image path to image id in coco style
  int image_id = -1;
  try {
    image_id = stoi(image_name);
  } catch (std::invalid_argument &e) {
    // maybe not a coco val image.
  }
  std::lock_guard<std::mutex> lk(mtx_);
  for (const auto &person : persons)
    Write(img, image_id, keypoints, person);
}

void COCOResultSaver::Write(
    cv::Mat img,
    int image_id,
    const Keypoints &keypoints,
    const PersonInfo &person) {
  // keypoints order in coco format
  const auto &indexes_in_coco_order = GetNetParams().indexes_in_coco_order;

  static constexpr int category_id = 1;
  writer_.StartObject();
  writer_.Key("image_id");
  writer_.Int(image_id);
  writer_.Key("category_id");
  writer_.Int(category_id);
  writer_.Key("keypoints");
  writer_.StartArray();
  for (int i : indexes_in_coco_order) {
    const auto &keypoint_idx = person.keypoint_idxs[i];
    if (kInvalidPos == keypoint_idx) {
      writer_.Int(-1);
      writer_.Int(-1);
      writer_.Int(0);
    } else {
      auto point = RoundPoint(keypoints[keypoint_idx.x][keypoint_idx.y].pos, img);
      writer_.Int(point.x);
      writer_.Int(point.y);
      writer_.Int(1);
    }
  }
  writer_.EndArray();
  writer_.Key("score");
  // keep 5 decimal places, otherwise the writing may fail
  CHECK_EQ(true, writer_.Double(static_cast<int>(person.score * 100000) / 100000.0));
  writer_.EndObject();
}

