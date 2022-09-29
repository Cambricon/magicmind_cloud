#include <cnrt.h>
#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "../include/post_process.hpp"
#include "../include/pre_process.hpp"
#include "../include/utils.hpp"

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "",
              "The image directory");  //"./../../../datasets/coco/test";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch, 1, "The batch size");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 1. cnrt init
  std::cout << "Cnrt init..." << std::endl;
  MluDeviceGuard device_guard(FLAGS_device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);

  // 2.create model
  std::cout << "Load model..." << std::endl;
  auto model = CreateIModel();
  CHECK_PTR(model);
  MM_CHECK(model->DeserializeFromFile(FLAGS_magicmind_model.c_str()));
  PrintModelInfo(model);

  // 3. crete engine
  std::cout << "Create engine..." << std::endl;
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);

  // 4. create context
  std::cout << "Create context..." << std::endl;
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5. crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. memory alloc
  // input tensor mlu ptrs
  void *input_mlu_addr_ptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = FLAGS_batch;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  CNRT_CHECK(cnrtMalloc(&input_mlu_addr_ptr, input_tensors[0]->GetSize()));
  MM_CHECK(input_tensors[0]->SetData(input_mlu_addr_ptr));
  // output tensor mlu ptrs
  void *output_mlu_addr_ptr = nullptr;
  if (magicmind::Status::OK() ==
      context->InferOutputShape(input_tensors, output_tensors)) {
    for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id) {
      CNRT_CHECK(cnrtMalloc(&output_mlu_addr_ptr,
                            output_tensors[output_id]->GetSize()));
      MM_CHECK(output_tensors[output_id]->SetData(output_mlu_addr_ptr));
    }
  } else {
    std::cout << "InferOutputShape failed" << std::endl;
  }

  // output tensor cpu ptrs
  float *data_ptr = nullptr;
  data_ptr = new float[output_tensors[0]->GetSize() /
                       sizeof(output_tensors[0]->GetDataType())];
  int detection_num;
  vector<vector<float>> results;

  // 7. load image
  std::cout << "================== Load Images ===================="
            << std::endl;
  std::vector<std::string> image_paths = LoadImages(
      FLAGS_image_dir, FLAGS_batch, FLAGS_image_num, FLAGS_file_list);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir
              << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  std::cout << "Total images : " << image_num << std::endl;
  std::cout << "Start run..." << std::endl;
  for (int i = 0; i < image_num; i++) {
    string image_name =
        image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
    std::cout << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/"
              << image_num << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    cv::Mat img_pro = process_img(img);
    // copy in
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data,
                          input_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    CNRT_CHECK(
        cnrtMemcpy((void *)&detection_num, output_tensors[1]->GetMutableData(),
                   output_tensors[1]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy((void *)data_ptr, output_tensors[0]->GetMutableData(),
                          output_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_DEV2HOST));
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
    map<int, string> name_map = load_name(FLAGS_label_path);
    post_process(img, results, name_map, image_name, FLAGS_output_dir,
                 FLAGS_save_img);
    results.clear();
  }

  // 10. destroy resource
  delete[] data_ptr;
  for (auto tensor : input_tensors) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors) {
    if (output_mlu_addr_ptr != nullptr) {
      cnrtFree(tensor->GetMutableData());
    }
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
