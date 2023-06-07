#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <sys/stat.h>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>

#include "pre_process.hpp"
#include "post_process.hpp"
#include "utils.hpp"
#include "model_runner.h"

using namespace magicmind;

#define MINVALUE(A,B) ( A < B ? A : B )

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");  //"./../../../datasets/coco/test";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "coco_file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch_size, 1, "The batch size");

/**
 * @brief get detection box from model's output
 *        see mm_network.h IDetectionOutputNode for details
 * @param[out] results
 * @param[in] detection_num
 * @param[in] data_ptr
 */
void Yolov5GetBox(std::vector<std::vector<float>> &results, int detection_num, float *data_ptr) {
  for (int i = 0; i < detection_num; ++i) {
    std::vector<float> result;
    float class_idx = *(data_ptr + 7 * i + 1);
    float score = *(data_ptr + 7 * i + 2);
    float xmin = *(data_ptr + 7 * i + 3);
    float ymin = *(data_ptr + 7 * i + 4);
    float xmax = *(data_ptr + 7 * i + 5);
    float ymax = *(data_ptr + 7 * i + 6);

    result.push_back(class_idx);
    result.push_back(score);
    result.push_back(xmin);
    result.push_back(ymin);
    result.push_back(xmax);
    result.push_back(ymax);
    results.push_back(result);
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_yolov7("infer yolov7");

  // create an instance of ModelRunner
  auto yolov7_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!yolov7_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init yolov7 runnner failed.";
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
  if(FLAGS_image_num >= 0){
    image_num = MINVALUE(FLAGS_image_num, image_num);
  }
  size_t rem_image_num = image_num % FLAGS_batch_size;
  SLOG(INFO) << "Total images : " << image_num;
  // load label
  std::map<int, std::string> name_map = load_name(FLAGS_label_path);

  // batch information
  int batch_counter = 0;
  std::vector<std::string> batch_image_name;
  std::vector<cv::Mat> batch_image;

  // allocate host memory for batch preprpcessed data
  auto batch_data = yolov7_runner->GetHostInputData();

  // one batch input data addr offset
  int batch_image_offset = yolov7_runner->GetInputSizes()[0] / FLAGS_batch_size;

  auto input_dim = yolov7_runner->GetInputDims()[0];
  int h = input_dim[1];
  int w = input_dim[2];
  SLOG(INFO) << "img_h: " << h;
  SLOG(INFO) << "img_w: " << w;

  SLOG(INFO) << "Start run...";
  for (int i = 0; i < image_num; i++) {
    std::string image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
    std::cout << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/" << image_num
              << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    cv::Mat img_pro = process_img(img, h, w);
    batch_image_name.push_back(image_name);
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
      yolov7_runner->H2D();
      // compute
      yolov7_runner->Run(FLAGS_batch_size);
      // copy out
      yolov7_runner->D2H();
      // get model's output addr in host
      auto host_output_ptr = yolov7_runner->GetHostOutputData();
      // yolov7 model has two outputs
      auto detection_box = (float *)host_output_ptr[0];
      auto detection_num = (int *)host_output_ptr[1];

      magicmind::DataType box_dtype = yolov7_runner->GetOutputDataTypes()[0];
      // one batch detection box data addr offset
      int batch_box_offset =
          yolov7_runner->GetOutputSizes()[0] / FLAGS_batch_size / sizeof(box_dtype);

      for (int j = 0; j < real_batch_size; j++) {
        std::vector<std::vector<float>> results;
        // gets boxes from model's output.
        Yolov5GetBox(results, detection_num[j], detection_box + j * batch_box_offset);
        // postprocess
        post_process(batch_image[j], results, name_map, batch_image_name[j], FLAGS_output_dir,
                     FLAGS_save_img, h, w);
        results.clear();
      }
      batch_counter = 0;
      batch_image.clear();
      batch_image_name.clear();
    }
  }
  // destroy resource
  yolov7_runner->Destroy();
  return 0;
}
