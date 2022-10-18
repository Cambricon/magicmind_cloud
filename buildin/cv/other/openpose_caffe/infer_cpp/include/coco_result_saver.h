#ifndef COCO_RESULT_SAVER_H_
#define COCO_RESULT_SAVER_H_

#include <opencv2/opencv.hpp>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "utils.h"

struct StreamImpl {
  using Ch = char;
  std::ofstream ofs;
  void Put(Ch ch) { ofs.put(ch); }
  void Flush() { ofs.flush(); }
};

/**
 * Save keypoints in COCO style JSON format.
 * This is a thread-safe implemention.
 */
class COCOResultSaver {
 public:
  /**
   * @param output_path[in]: output file path
   */
  explicit COCOResultSaver(const std::string &output_path);
  ~COCOResultSaver();
  /**
   * @param img[in]: origin image
   * @param image_path[in]: image path
   * @param keypoints[in]: detected keypoints
   * @param persons[in]: detected persons
   */
  void Write(cv::Mat img, const std::string &image_path,
             const Keypoints &keypoints, const PersonInfos &persons);

 private:
  // image_id : COCO style image id, gets from image_path
  void Write(cv::Mat img, int image_id, const Keypoints &keypoints, const PersonInfo &person);

 private:
  std::mutex mtx_;
  StreamImpl stream_;
  rapidjson::Writer<StreamImpl> writer_;
};  // class COCOResultSaver

#endif  // COCO_RESULT_SAVER_H_

