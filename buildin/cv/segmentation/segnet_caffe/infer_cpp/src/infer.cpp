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

#include "post_process.h"
#include "pre_process.h"
#include "utils.h"

using namespace magicmind;
using namespace cv;

/**
 * @brief input params
 * magicmind_model: Magicmind model path;
 * image_dir: input images path;
 * name_file: label of image;
 * output_dir: the detection output path,include *.jpg;
 */
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_string(image_list, "", "The image file list");
DEFINE_string(output_dir, "", "The classification results output file");
DEFINE_bool(save_txt, true, "save txt");

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

  // 6. memory alloc
  // mlu
  void *mlu_input_addr_ptr = nullptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = 1;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  CNRT_CHECK(cnrtMalloc(&mlu_input_addr_ptr, input_tensors[0]->GetSize()));
  CHECK_STATUS(input_tensors[0]->SetData(mlu_input_addr_ptr));

  void *mlu_output_addr_ptr = nullptr;
  auto output_num = model->GetOutputNum();
  if (magicmind::Status::OK() ==
      context->InferOutputShape(input_tensors, output_tensors)) {
    std::cout << "InferOutputShape sucessed" << std::endl;
    for (size_t output_id = 0; output_id < output_num; ++output_id) {
      CNRT_CHECK(cnrtMalloc(&mlu_output_addr_ptr,
                            output_tensors[output_id]->GetSize()));
      CHECK_STATUS(output_tensors[output_id]->SetData(mlu_output_addr_ptr));
    }
  } else {
    std::cout << "InferOutputShape failed" << std::endl;
  }

  // cpu
  void *output_cpu_ptrs = (void *)malloc(output_tensors[0]->GetSize());

  // 7. load image
  std::cout << "================== Load Images ===================="
            << std::endl;
  std::vector<std::string> image_paths =
      LoadImages(FLAGS_image_dir, FLAGS_image_list, input_dim[0]);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir
              << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  std::cout << "Total images : " << image_num << std::endl;
  std::cout << "Start run..." << std::endl;
  for (int i = 0; i < image_num; i++) {
    std::string image_name = GetFileName(image_paths[i]);
    std::cout << "Inference img: " << image_name << "\t\t\t" << i << "/"
              << image_num << std::endl;

    Mat img = imread(image_paths[i]);
    Mat img_pre = Preprocess(img, input_dim[1], input_dim[2]);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pre.data,
                          input_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    // output_tensors.clear();
    CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    std::vector<std::vector<float>> results;

    for (uint32_t j = 0; j < output_num; j++) {
      int detection_num = output_tensors[j]->GetDimensions()[0];
      // get output data from tensor
      auto preds_dim = output_tensors[j]->GetDimensions();
      std::shared_ptr<void> preds_ptr = nullptr;
      cv::Mat pred;
      if (output_tensors[j]->GetMemoryLocation() ==
          magicmind::TensorLocation::kHost) {
        // memory in host
        pred = PostProcess(img, preds_dim,
                           (float *)output_tensors[j]->GetMutableData());
      } else if (output_tensors[j]->GetMemoryLocation() ==
                 magicmind::TensorLocation::kMLU) {
        // memory in device
        CHECK_CNRT(cnrtMemcpy, output_cpu_ptrs,
                   output_tensors[j]->GetMutableData(),
                   output_tensors[j]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST);
        pred = PostProcess(img, preds_dim, (float *)output_cpu_ptrs);
      } else {
        std::cout << "Invalid memory location." << std::endl;
      }
      if (FLAGS_save_txt) {
        std::string save_path =
            FLAGS_output_dir + "/" + image_name + "_result.binary";
        std::ofstream ofs(save_path, std::ios::binary);
        if (!ofs.is_open()) {
          std::cout << "Create file [" << save_path << "] failed." << std::endl;
        }
        ofs.write((char *)pred.data, pred.cols * pred.rows);
        ofs.close();
      }
    }
  }

  // 9. destroy resource
  for (auto tensor : input_tensors) {
    cnrtFree(tensor->GetMutableData());
    tensor->Destroy();
  }
  for (auto tensor : output_tensors) {
    if (mlu_output_addr_ptr != nullptr) {
      cnrtFree(tensor->GetMutableData());
    }
    tensor->Destroy();
  }
  context->Destroy();
  engine->Destroy();
  model->Destroy();
  return 0;
}
