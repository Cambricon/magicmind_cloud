#include <mm_runtime.h>
#include <cnrt.h>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pre_process.h"
#include "utils.hpp"
#include "model_runner.h"
#include "logger.h"

using namespace magicmind;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "the magicmind model path");
DEFINE_string(image_dir, "", "predict image file path");
DEFINE_string(image_list, "", "predict image list");
DEFINE_string(output_dir, "", "output path");
DEFINE_int32(image_num, 1000, "image num");
DEFINE_bool(save_img, false, "save img or not. default: false");
DEFINE_int32(batch_size, 1, "The batch size");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_arcface("infer arcface");

  // create an instance of ModelRunner
  auto arcface_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!arcface_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init arcface runnner failed.";
    return false;
  }

  // load image and label
  std::cout << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths =
      LoadImages(FLAGS_image_dir, FLAGS_image_list, FLAGS_image_num, FLAGS_batch_size);
  if (image_paths.size() == 0) {
    SLOG(INFO) << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  size_t rem_image_num = image_num % FLAGS_batch_size;
  SLOG(INFO) << "Total images : " << image_num << std::endl;
  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;
  std::vector<std::string> faceness_scores;

  // allocate host memory for batch preprpcessed data
  auto batch_data = arcface_runner->GetHostInputData();

  // one batch input data addr offset
  int batch_image_offset = arcface_runner->GetInputSizes()[0] / FLAGS_batch_size;
  auto input_dim = arcface_runner->GetInputDims()[0];

  SLOG(INFO) << "Start run...";
  for (int i = 0; i < image_num; i++) {
    auto line = image_paths[i];
    std::vector<std::string> res = SplitString(line);
    std::string path = FLAGS_image_dir + "/" + res[0];
    std::string image_name = GetFileName(path);
    batch_image_name.emplace_back(image_name);
    int end_pos = res[0].find('.');
    std::string id = res[0].substr(0, end_pos);
    std::string faceness_score = res[11];
    faceness_scores.emplace_back(faceness_score);
    std::vector<std::string> landmarks;
    landmarks.insert(landmarks.end(), res.begin() + 1, res.end() - 1);
    if (!check_file_exist(path)) {
      SLOG(INFO) << "image file " + path + " not found.\n";
      exit(1);
    }
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
      SLOG(INFO) << "Failed to open image file " + image_paths[i];
      exit(1);
    }
    SLOG(INFO) << "Inference img: " << image_name << "\t\t\t" << i << "/" << image_num << std::endl;
    cv::Mat img_pro = Preprocess(img, input_dim, landmarks);
    batch_image.push_back(img);
    memcpy((u_char *)(batch_data[0]) + batch_counter * batch_image_offset, img_pro.data,
           batch_image_offset);
    batch_counter += 1;
    size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
    if (batch_counter % real_batch_size == 0) {
      // copy in
      arcface_runner->H2D();
      // compute
      arcface_runner->Run(FLAGS_batch_size);
      // copy out
      arcface_runner->D2H();
      // get model's output addr in host
      auto host_output_ptr = arcface_runner->GetHostOutputData()[0];

      if (FLAGS_save_img) {
        auto output_dim = arcface_runner->GetOutputDims()[0];
        for (int j = 0; j < real_batch_size; j++) {
          std::string save_path = FLAGS_output_dir + "/" + batch_image_name[j] + ".feature";
          std::ofstream ofs(save_path);
          for (int k = 0; k < output_dim[1]; k++) {
            ofs << ((float *)host_output_ptr)[k + j * output_dim[1]] << " ";
          }
          ofs << faceness_scores[j] << std::endl;
          ofs.close();
        }
      }
      batch_counter = 0;
      batch_image_name.clear();
      faceness_scores.clear();
    }
  }
  // destroy resource
  arcface_runner->Destroy();
  return 0;
}
