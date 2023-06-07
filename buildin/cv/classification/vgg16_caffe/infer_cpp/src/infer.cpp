#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "utils.hpp"
#include "pre_process.hpp"
#include "post_process.hpp"
#include "model_runner.h"

using namespace std;
using namespace cv;

#define MINVALUE(A, B) (A < B ? A : B)

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(image_dir, "", "dataset_dir");
DEFINE_string(label_file, "labels.txt", "labels.txt");
DEFINE_string(name_file, "name.txt", "The label path");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "", "The classification results label output file");
DEFINE_string(result_top1_file, "", "The classification results top1 output file");
DEFINE_string(result_top5_file, "", "The classification results top5 output file");
DEFINE_int32(image_num, 0, "image number");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_vgg16_caffe("infer vgg16_caffe");

  // create an instance of ModelRunner
  auto vgg16_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!vgg16_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init vgg16 runnner failed.";
    return false;
  }

  // load images
  SLOG(INFO) << "================== Load Images ====================";
  result *test;
  test = LoadImages(FLAGS_image_dir, FLAGS_batch_size, FLAGS_label_file);
  if (test->image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support JPEG jpg.";
    return 0;
  }

  int img_w = 224, img_h = 224, img_c = 3, n_classes = 1000;
  size_t image_num = test->image_paths.size();
  std::map<int, std::string> name_map = load_name(FLAGS_name_file);

  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;
  std::vector<int> batch_image_label;

  Record result_label(FLAGS_result_label_file);
  Record result_top1_file(FLAGS_result_top1_file);
  Record result_top5_file(FLAGS_result_top5_file);
  Record result_file(FLAGS_result_file);

  // allocate host memory for batch preprpcessed data
  auto batch_data = vgg16_runner->GetHostInputData();
  // one batch input data addr offset
  int batch_image_offset = vgg16_runner->GetInputSizes()[0] / FLAGS_batch_size;

  SLOG(INFO) << "Start run...";

  // cambricon-note: if FLAGS_image_num is not set,the image_num is:test->image_paths.size()
  // if set to a positive num, the image_num is the same as FLAGS_image_num
  // if set to a negative num, the image_num is the same as test->image_paths.size()
  if (FLAGS_image_num > 0) {
    image_num = MINVALUE(FLAGS_image_num, image_num);
  }
  size_t rem_image_num = image_num % FLAGS_batch_size;
  SLOG(INFO) << "Total images : " << image_num;

  for (int i = 0; i < image_num; i++) {
    std::string image_name =
        test->image_paths[i].substr(test->image_paths[i].find_last_of('/') + 1, 23);
    int image_label = test->labels[i];
    std::cout << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/" << image_num;
    cv::Mat src_img = cv::imread(test->image_paths[i]);
    // cv::Mat pro_img (img_h, img_w, CV_8UC3);
    // Preprocess(src_img , img_h , img_w , pro_img);
    cv::Mat pro_img = Preprocess(src_img, img_h, img_w);
    batch_image_name.push_back(image_name);
    batch_image_label.push_back(image_label);

    // batching preprocessed data
    memcpy((u_char *)(batch_data[0]) + batch_counter * batch_image_offset, pro_img.data,
           batch_image_offset);

    batch_counter += 1;
    // image_num may not be divisible by FLAGS_batch.
    // real_batch_size records number of images in every loop, real_batch_size may change in the
    // last loop.
    size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
    std::cout << "real_batch_size: " << real_batch_size << std::endl;
    if (batch_counter % real_batch_size == 0) {
      // copy in
      vgg16_runner->H2D();
      // compute
      vgg16_runner->Run(FLAGS_batch_size);
      // copy out
      vgg16_runner->D2H();
      // get model's output addr in host
      auto host_output_ptr = vgg16_runner->GetHostOutputData();

      auto infer_res = (float *)host_output_ptr[0];

      // post process
      for (int j = 0; j < real_batch_size; j++) {
        if (!FLAGS_result_label_file.empty()) {
          result_label.write("[" + std::to_string(i + 1 - real_batch_size + j) +
                                 "]: " + std::to_string(batch_image_label[j]),
                             false);
        }
        std::vector<int> top5 = ArgTopK(infer_res + j * n_classes, n_classes, 5);
        if (!FLAGS_result_top1_file.empty()) {
          result_top1_file.write(
              "[" + std::to_string(i + 1 - real_batch_size + j) + "]: " + std::to_string(top5[0]),
              false);
        }
        result_file.write("top5 result in " + batch_image_name[j] + ":", false);
        for (int k = 0; k < 5; k++) {
          if (!FLAGS_result_top5_file.empty()) {
            result_top5_file.write(
                "[" + std::to_string(i + 1 - real_batch_size + j) + "]: " + std::to_string(top5[k]),
                false);
          }
          if (!FLAGS_result_file.empty()) {
            result_file.write(std::to_string(k) + " [" + name_map[top5[k]] + "]", false);
          }
        }
      }
      batch_counter = 0;
      batch_image_name.clear();
      batch_image_label.clear();
    }
  }

  // destroy resource
  vgg16_runner->Destroy();
  return 0;
}
