#include <cnrt.h>
#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <sys/stat.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "../include/post_process.h"
#include "../include/pre_process.h"
#include "../include/utils.h"

using namespace magicmind;
using namespace cv;

/**
 * @brief input params
 * magicmind_model: Magicmind model path;
 * image_dir: input images path;
 * output_dir: the detection output path,include *.jpg;
 */

DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_string(output_dir, "", "The classification results output file");
DEFINE_string(file_list, "", "The image file list");
DEFINE_int32(image_num, 1000, "image number");
DEFINE_string(shape_mutable, "true", "dynamic or static h/w");
DEFINE_bool(save_img, true, "save img");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 1. cnrt init
  std::cout << "Cnrt init..." << std::endl;
  uint8_t device_id = 0;
  MluDeviceGuard device_guard(device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);

  // 2. create model
  std::cout << "Load model..." << std::endl;
  auto model = CreateIModel();
  CHECK_PTR(model);
  CHECK_STATUS(model->DeserializeFromFile(FLAGS_magicmind_model.c_str()));
  PrintModelInfo(model);

  // 3. crete engine
  std::cout << "Create engine..." << std::endl;
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);

  // 4. create context
  std::cout << "Create context..." << std::endl;
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5. crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. input output tensor memory alloc
  void *input_mlu_ptrs;
  void *output_mlu_ptrs;
  auto input_dim = model->GetInputDimension(0);
  auto output_dim = model->GetOutputDimension(0);

  int input_size = 513;
  CNRT_CHECK(cnrtMalloc(&input_mlu_ptrs,
                        input_dim[0] * input_size * input_size * input_dim[3]));
  CHECK_STATUS(input_tensors[0]->SetData(input_mlu_ptrs));
  CNRT_CHECK(
      cnrtMalloc(&output_mlu_ptrs, output_dim[0] * input_size * input_size));
  CHECK_STATUS(output_tensors[0]->SetData(output_mlu_ptrs));
  uint32_t *output_cpu_ptrs =
      new uint32_t[output_dim[0] * input_size * input_size];

  // 7. load image
  std::cout << "================== Load Images ===================="
            << std::endl;
  std::vector<cv::String> image_paths = LoadImages(
      FLAGS_image_dir, input_dim[0], FLAGS_image_num, FLAGS_file_list);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir
              << "]. Support jpg." << std::endl;
    return 0;
  }
  size_t image_num = image_paths.size();
  std::cout << "Total images : " << image_num << std::endl;
  std::cout << "Start run..." << std::endl;
  for (int i = 0; i < image_num; i++) {
    std::string image_name = GetFileName(image_paths[i]);
    std::cout << "Inference img: " << image_name << ".jpg"
              << "\t\t\t\t" << i << "/" << image_num << std::endl;
    Mat img = imread(image_paths[i]);
    Mat img_pre = Preprocess(img, FLAGS_shape_mutable);

    input_tensors[0]->SetDimensions(
        Dims({input_dim[0], img_pre.rows, img_pre.cols, input_dim[3]}));
    if (magicmind::Status::OK() ==
        context->InferOutputShape(input_tensors, output_tensors)) {
      std::cout << "InferOutputShape sucessed" << std::endl;
      CHECK_STATUS(output_tensors[0]->SetData(output_mlu_ptrs));
    } else {
      std::cout << "InferOutputShape failed" << std::endl;
    }

    if (input_tensors[0]->GetMemoryLocation() ==
        magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pre.data,
                            input_tensors[0]->GetSize(),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    // 8. compute
    CHECK_STATUS(context->Enqueue(input_tensors, &output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    if (output_tensors[0]->GetMemoryLocation() ==
        magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMemcpy(
          output_cpu_ptrs, output_tensors[0]->GetMutableData(),
          output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      PostProcess((uint32_t *)output_cpu_ptrs, img_pre.rows, img_pre.cols,
                  image_name, FLAGS_output_dir, FLAGS_save_img);
    } else {
      PostProcess((uint32_t *)output_tensors[0]->GetMutableData(), img_pre.rows,
                  img_pre.cols, image_name, FLAGS_output_dir, FLAGS_save_img);
    }
  }

  // 9. destroy resource
  delete[] output_cpu_ptrs;
  for (auto tensor : input_tensors) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors) {
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
