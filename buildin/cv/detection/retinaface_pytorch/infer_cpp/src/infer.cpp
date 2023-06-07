#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>

#include "../include/pre_process.hpp"
#include "../include/post_process.hpp"
#include "utils.hpp"
#include "model_runner.h"

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");  //"./../../../datasets/widerface/WIDER_val";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "", "file_list");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch_size, 1, "The batch size");

// write detection results to file
void WritePreds(const std::string &img_path, const std::vector<BBox> &bboxes) {
  auto dot_pos = img_path.rfind('.');
  auto filename = GetFileName(img_path);
  std::string save_path = img_path.substr(0, dot_pos + 1) + "txt";
  std::ofstream ofs(save_path);
  if (!ofs.is_open()) {
    SLOG(ERROR) << "ofstream is not opened";
    return;
  }
  ofs << filename << std::endl;
  ofs << bboxes.size() << std::endl;
  for (auto bbox : bboxes) {
    ofs << static_cast<int>(std::round(bbox.cx - bbox.w / 2.0f)) << ' '
        << static_cast<int>(std::round(bbox.cy - bbox.h / 2.0f)) << ' ' << bbox.w << ' ' << bbox.h
        << ' ' << bbox.score << std::endl;
  }
  ofs.close();
}

// draw bboxes on image
void Draw(cv::Mat img, const std::vector<BBox> &bboxes) {
  for (const auto &bbox : bboxes) {
    cv::Point p1, p2;
    p1.x = std::max(0, bbox.cx - static_cast<int>(std::floor(bbox.w / 2.f)));
    p1.y = std::max(0, bbox.cy - static_cast<int>(std::floor(bbox.h / 2.f)));
    p2.x = std::min(img.cols, bbox.cx + static_cast<int>(std::floor(bbox.w / 2.f)));
    p2.y = std::min(img.rows, bbox.cy + static_cast<int>(std::floor(bbox.h / 2.f)));
    cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 0), 1);
    // landmarks
    for (auto landm : bbox.landms) {
      cv::circle(img, landm, 1, cv::Scalar(0, 0, 255), 2);
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_retinaface("infer retinaface");

  // create an instance of ModelRunner
  auto retinaface_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!retinaface_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init retinaface runnner failed.";
    return false;
  }

  // load image
  std::cout << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths =
      LoadImages(FLAGS_image_dir, FLAGS_batch_size, FLAGS_image_num, FLAGS_file_list);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  size_t rem_image_num = image_num % FLAGS_batch_size;
  std::cout << "Total images: " << image_num << std::endl;

  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;

  // allocate host memory for batch preprpcessed data
  auto batch_data = retinaface_runner->GetHostInputData();

  // one batch input data addr offset
  int batch_image_offset = retinaface_runner->GetInputSizes()[0] / FLAGS_batch_size;

  auto input_dim = retinaface_runner->GetInputDims()[0];
  int h = input_dim[1];
  int w = input_dim[2];
  SLOG(INFO) << "Start run...";
  for (int i = 0; i < image_num; i++) {
    string image_name = GetFileName(image_paths[i]);
    std::cout << "Inference img: " << image_name << "\t\t\t" << i + 1 << "/" << image_num
              << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    cv::Mat img_pro = Preprocess(img, h, w);
    batch_image_name.push_back(image_paths[i]);
    batch_image.push_back(img);

    // batching preprocessed data
    memcpy((u_char *)(batch_data[0]) + batch_counter * batch_image_offset, img_pro.data,
           batch_image_offset);

    batch_counter += 1;
    // image_num may not be divisible by FLAGS_batch.
    // real_batch_size records number of images in every loop, real_batch_size may change in the
    // last loop.
    size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
    if (batch_counter % real_batch_size == 0) {
      // copy in
      retinaface_runner->H2D();
      // compute
      retinaface_runner->Run(FLAGS_batch_size);
      // copy out
      retinaface_runner->D2H();
      // get model's output addr in host
      auto host_output_ptr = retinaface_runner->GetHostOutputData();
      // retinaface model has three outputs
      std::vector<const float *> results;

      auto detection_loc = (float *)host_output_ptr[0];
      auto detection_conf = (float *)host_output_ptr[1];
      auto detection_landms = (float *)host_output_ptr[2];

      magicmind::DataType box_dtype = retinaface_runner->GetOutputDataTypes()[0];
      // one batch detection box data addr offset
      int batch_loc_offset =
          retinaface_runner->GetOutputSizes()[0] / FLAGS_batch_size / sizeof(box_dtype);
      int batch_conf_offset =
          retinaface_runner->GetOutputSizes()[1] / FLAGS_batch_size / sizeof(box_dtype);
      int batch_landms_offset =
          retinaface_runner->GetOutputSizes()[2] / FLAGS_batch_size / sizeof(box_dtype);
      auto output_dims = retinaface_runner->GetOutputDims();
      for (int j = 0; j < real_batch_size; j++) {
        // gets boxes from model's output.
        results.push_back(detection_loc + j * batch_loc_offset);
        results.push_back(detection_conf + j * batch_conf_offset);
        results.push_back(detection_landms + j * batch_landms_offset);

        std::vector<BBox> bboxes = Postprocess(batch_image[j], input_dim, results, output_dims);
        char abs_path[PATH_MAX] = {0};
        if (realpath(FLAGS_image_dir.c_str(), abs_path) == NULL) {
          std::cout << "Get real image path in " << FLAGS_image_dir.c_str() << " failed...";
          exit(1);
        }
        std::string image_dir(abs_path);
        std::string image_relative_path = batch_image_name[j].substr(image_dir.length());
        auto slash_pos = image_relative_path.rfind("/");
        if (slash_pos != std::string::npos) {
          std::string mkdir_cmd = "mkdir -p " + FLAGS_output_dir + "/pred_txts/" +
                                  image_relative_path.substr(0, slash_pos);
          system(mkdir_cmd.c_str());
          mkdir_cmd = "mkdir -p " + FLAGS_output_dir + "/images/" +
                      image_relative_path.substr(0, slash_pos);
          system(mkdir_cmd.c_str());
        }
        WritePreds(FLAGS_output_dir + "/pred_txts/" + image_relative_path, bboxes);
        if (FLAGS_save_img) {
          Draw(batch_image[j], bboxes);
          cv::imwrite(FLAGS_output_dir + "/images/" + image_relative_path, batch_image[j]);
        }
        results.clear();
      }
      batch_counter = 0;
      batch_image.clear();
      batch_image_name.clear();
    }
  }
  // destroy resource
  retinaface_runner->Destroy();
  return 0;
}
