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
#include "pre_process.h"
#include "post_process.h"
#include "model_runner.h"

using namespace std;
using namespace cv;

#define MINVALUE(A, B) (A < B ? A : B)

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(image_dir, "", "dataset_dir");
DEFINE_string(image_list, "", "The image file list");
DEFINE_int32(image_num, 0, "image number");
DEFINE_string(output_dir, "", "The dir used for saving infer results ");
DEFINE_bool(save_txt, true, "save txt");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_segnet_caffe("infer segnet_caffe");

  // create an instance of ModelRunner
  auto cur_model_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!cur_model_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init cur_model_runner failed.";
    return false;
  }

  // load images
  SLOG(INFO) << "================== Load Images ====================";
  std::vector<std::string> image_paths =
      LoadImages(FLAGS_image_dir, FLAGS_image_list, FLAGS_batch_size);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
    return 0;
  }

  // nchw
  auto input_dims = cur_model_runner->GetInputDims();
  int img_h = input_dims[0][1];
  int img_w = input_dims[0][2];
  SLOG(INFO) << "======= img_h: " << img_h;
  SLOG(INFO) << "======= img_w: " << img_w;

  size_t image_num = image_paths.size();

  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;

  // allocate host memory for batch preprpcessed data
  auto batch_data = cur_model_runner->GetHostInputData();
  // one batch input data addr offset
  int batch_image_offset = cur_model_runner->GetInputSizes()[0] / FLAGS_batch_size;

  SLOG(INFO) << "Start run...";

  // cambricon-note: if FLAGS_image_num is not set,the image_num is:test->image_paths.size()
  // if set to a positive num, the image_num is the same as FLAGS_image_num
  // if set to a negative num, the image_num is the same as image_paths.size()
  if (FLAGS_image_num > 0) {
    image_num = MINVALUE(FLAGS_image_num, image_num);
  }
  size_t rem_image_num = image_num % FLAGS_batch_size;
  SLOG(INFO) << "Total images : " << image_num;

  for (int i = 0; i < image_num; i++) {
    std::string tmp_image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 23);
    std::string image_name = tmp_image_name.substr(0, tmp_image_name.find_last_of('.'));
    SLOG(INFO) << "Inference img : " << image_name << ".jpg\t\t\t" << i + 1 << "/" << image_num;
    cv::Mat src_img = cv::imread(image_paths[i]);
    cv::Mat pro_img = Preprocess(src_img, img_h, img_w);
    batch_image_name.push_back(image_name);

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
      cur_model_runner->H2D();
      // compute
      cur_model_runner->Run(FLAGS_batch_size);
      // copy out
      cur_model_runner->D2H();
      // get model's output addr in host
      auto host_output_ptr = cur_model_runner->GetHostOutputData();
      auto infer_res = (float *)host_output_ptr[0];

      auto output_tensors = cur_model_runner->GetOutputTensors();
      auto output_size = cur_model_runner->GetOutputSizes();
      int output_num = output_size.size();
      auto output_ptrs = cur_model_runner->OutputPtrs();

      // post process
      for (int i = 0; i < real_batch_size; i++) {
        for (int j = 0; j < output_num; j++) {
          auto preds_dim = output_tensors[j]->GetDimensions();
          cv::Mat pred;
          pred = PostProcess(src_img, preds_dim, infer_res);
          if (FLAGS_save_txt) {
            std::string save_path = FLAGS_output_dir + "/" + image_name + "_result.binary";
            std::ofstream ofs(save_path, std::ios::binary);
            if (!ofs.is_open()) {
              std::cout << "Create file [" << save_path << "] failed." << std::endl;
            }
            ofs.write((char *)pred.data, pred.cols * pred.rows);
            ofs.close();
          }
        }

      }  // end for
      batch_counter = 0;
      batch_image_name.clear();
    }
  }

  // destroy resource
  cur_model_runner->Destroy();
  return 0;
}
