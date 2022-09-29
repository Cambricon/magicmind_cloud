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
DEFINE_string(image_dir, "", "The image directory");
DEFINE_int32(image_num, 1, "image number");
DEFINE_string(file_list, "file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_int32(max_bbox_num, 100, "Max number of bounding-boxes per image");
DEFINE_double(confidence_thresholds, 0.001, "");
DEFINE_string(output_dir, "", "../data/images");
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

  // 6. input tensor memory alloc
  void *mlu_addr_ptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = FLAGS_batch;
  }
  std::vector<magicmind::Dims> output_dims;
  for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id) {
    auto output_dim_vec = model->GetOutputDimension(output_id).GetDims();
    if (output_dim_vec[0] == -1) {
      output_dim_vec[0] = FLAGS_batch;
    }
    output_dims.push_back(magicmind::Dims(output_dim_vec));
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  MM_CHECK(input_tensors[0]->SetData(mlu_addr_ptr));
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
  std::vector<float *> net_outputs;
  for (size_t output_id = 0; output_id < output_dims.size(); output_id++) {
    float *data_ptr =
        new float[output_tensors[output_id]->GetSize() /
                  sizeof(output_tensors[output_id]->GetDataType())];
    net_outputs.push_back(data_ptr);
  }
  for (int i = 0; i < image_num; i++) {
    string image_name =
        image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
    std::cout << "Inference img : " << image_name << "\t\t\t" << i + 1 << "/"
              << image_num << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    cv::Mat img_pro = Preprocess(img, input_dim);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data,
                          input_tensors[0]->GetSize(),
                          CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    for (size_t output_id = 0; output_id < output_dims.size(); output_id++) {
      CNRT_CHECK(cnrtMemcpy(
          net_outputs[output_id], output_tensors[output_id]->GetMutableData(),
          output_tensors[output_id]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
    // postprocess
    map<int, string> name_map = load_name(FLAGS_label_path);
    auto bboxes = Postprocess(net_outputs, output_dims, FLAGS_max_bbox_num,
                              FLAGS_confidence_thresholds);
    // rescale bboxes to origin image.
    RescaleBBox(img, output_dims[0], bboxes, name_map, image_name,
                FLAGS_output_dir);
    if (FLAGS_save_img) {
      // draw bboxes on original image and save it to disk.
      cv::Mat origin_img = img.clone();
      Draw(img, bboxes, name_map);
      cv::imwrite(FLAGS_output_dir + "/" + image_name + ".jpg", img);
    }
  }

  // 10. destroy resource
  for (vector<float *>::const_iterator itr = net_outputs.begin();
       itr != net_outputs.end(); ++itr) {
    delete *itr;
  }
  net_outputs.clear();
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
