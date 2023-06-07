#include <cnrt.h>
#include <mm_runtime.h>

#include "../include/post_process.hpp"
#include "../include/pre_process.hpp"
#include "../include/utils.hpp"
using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(name_file, "/mm_ws/proj/datasets/imagenet/name.txt",
              "The label path");
DEFINE_string(label_file, "labels.txt", "labels.txt");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "",
              "The classification results label output file");
DEFINE_string(result_top1_file, "",
              "The classification results top1 output file");
DEFINE_string(result_top5_file, "",
              "The classification results top5 output file");
DEFINE_int32(batch_size, 1, "The batch size");

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
  // Check if current program can deal with this model
  if (!CheckModel(model)) {
    std::cout << "Can not deal with this model." << std::endl;
    std::cout
        << "You should provide a classification model and check the following:"
        << std::endl;
    std::cout << "1. Make sure the data type of input is UINT8." << std::endl;
    std::cout << "2. Make sure the input data is in NHWC order." << std::endl;
    std::cout << "3. Make sure the data type of output is FLOAT." << std::endl;
  }
  // 3. crete engine
  std::cout << "Create engine..." << std::endl;
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  auto engine = model->CreateIEngine(engine_config);
  CHECK_PTR(engine);

  // 4. create context
  std::cout << "Create context..." << std::endl;
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5. create input tensor and output tensor
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. memory malloc
  // mlu
  void *ptr = nullptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = FLAGS_batch_size;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  if (input_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
    CNRT_CHECK(cnrtMalloc(&ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(ptr));
  } 

  bool dynamic_output = false;
  if (magicmind::Status::OK() ==
      context->InferOutputShape(input_tensors, output_tensors)) {
    std::cout << "InferOutputShape successed" << std::endl;
    if (output_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CNRT_CHECK(cnrtMalloc(&ptr, output_tensors[0]->GetSize()));
      MM_CHECK(output_tensors[0]->SetData(ptr));
    }  
  } else {
      std::cout << "InferOutputShape failed" << std::endl;
      dynamic_output = true;
  }
  const int classes = 1000;
  const int elem_data_count = input_tensors[0]->GetSize() / sizeof(uint8_t) / FLAGS_batch_size;

  // cpu
  uint8_t *input_data_ptr = new uint8_t[FLAGS_batch_size * elem_data_count];
  float *output_data_ptr = new float[FLAGS_batch_size * classes];

  // 7. load image
  std::cout << "================== Load Images ===================="
            << std::endl;
  result *test;
  test = LoadImages(FLAGS_image_dir, FLAGS_batch_size, FLAGS_image_num,
                    FLAGS_label_file);
  if (test->image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir
              << "]. Support jpg.";
    return 0;
  }
  size_t image_num = test->image_paths.size();
  std::cout << "Total images : " << image_num << std::endl;
  std::map<int, std::string> name_map = load_name(FLAGS_name_file);

  Record result_label(FLAGS_result_label_file);
  Record result_top1_file(FLAGS_result_top1_file);
  Record result_top5_file(FLAGS_result_top5_file);
  Record result_file(FLAGS_result_file);
  vector<string> image_names(FLAGS_batch_size);
  vector<int> image_labels(FLAGS_batch_size);
  std::cout << "Start run..." << std::endl;
  for (int i = 0; i <= image_num; i++) {
    if (i != 0 && i % FLAGS_batch_size == 0) {
      // Memcpy H2D
      CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), input_data_ptr,
                            input_tensors[0]->GetSize(),
                            CNRT_MEM_TRANS_DIR_HOST2DEV));

      // 8. compute
      MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
      CNRT_CHECK(cnrtQueueSync(queue));

      // 9. copy out
      CNRT_CHECK(cnrtMemcpy(
          output_data_ptr, output_tensors[0]->GetMutableData(),
          output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));

      // post-process
      for (int _batch = 0; _batch < FLAGS_batch_size; _batch++) {
        std::string image_name = image_names[_batch].substr(
            image_names[_batch].find_last_of('/') + 1, 23);
        if (!FLAGS_result_label_file.empty()) {
          result_label.write("[" + std::to_string(i - FLAGS_batch_size + _batch) +
                                 "]: " + std::to_string(image_labels[_batch]),
                             false);
        }
        // get top5
        std::vector<int> top5 =
            ArgTopK(output_data_ptr + _batch * classes, classes, 5);
        // get FLAGS_result_top1_file
        if (!FLAGS_result_top1_file.empty()) {
          result_top1_file.write("[" +
                                     std::to_string(i - FLAGS_batch_size + _batch) +
                                     "]: " + std::to_string(top5[0]),
                                 false);
        }
        // get FLAGS_result_top5_file, FLAGS_result_file
        result_file.write("top5 result in " + image_name + ":", false);
        for (int j = 0; j < 5; j++) {
          if (!FLAGS_result_top5_file.empty()) {
            result_top5_file.write(
                "[" + std::to_string(i - FLAGS_batch_size + _batch) +
                    "]: " + std::to_string(top5[j]),
                false);
          }
          if (!FLAGS_result_file.empty()) {
            result_file.write(
                std::to_string(j) + " [" + name_map[top5[j]] + "]", false);
          }
        }
        if (i == image_num && _batch == (FLAGS_image_num % FLAGS_batch_size - 1))
          break;
      }
    }
    if (i == image_num) continue;
    std::string image_name = test->image_paths[i].substr(
        test->image_paths[i].find_last_of('/') + 1, 23);
    std::cout << "Inference img : " << test->image_paths[i] << "\t\t\t" << i + 1
              << "/" << image_num << std::endl;
    cv::Mat img = cv::imread(test->image_paths[i]);
    // pre-process
    if (img.data != NULL) {
      image_names[i % FLAGS_batch_size] = test->image_paths[i];
      image_labels[i % FLAGS_batch_size] = test->labels[i];
      cv::Mat dst_img = Preprocess(img, input_dim);
      memcpy(input_data_ptr + (i % FLAGS_batch_size) * elem_data_count, dst_img.data,
             elem_data_count * sizeof(uint8_t));
    }
  }

  // 10. destroy resource
  // destroy must do strictly as follow
  // destroy tensor/address first
  delete[] output_data_ptr;
  delete[] input_data_ptr;
  for (auto tensor : input_tensors) {
    if (tensor->GetMemoryLocation() == magicmind::TensorLocation::kMLU){
      cnrtFree(tensor->GetMutableData());
    }
    tensor->Destroy();
  }
  for (auto tensor : output_tensors) {
    if (!dynamic_output) {
      cnrtFree(tensor->GetMutableData());
    }
    tensor->Destroy();
  }
  // destroy context
  context->Destroy();
  // destory engine
  engine->Destroy();
  // destroy model
  model->Destroy();
  // destroy other
  cnrtQueueDestroy(queue);
  return 0;
}
