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
#include "../include/utils.hpp"

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory"); //"./../../../datasets/widerface/WIDER_val";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "", "file_list");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch, 1, "The batch size");

// write detection results to file
void WritePreds(const std::string &img_path, const std::vector<BBox> &bboxes) {
  auto dot_pos = img_path.rfind('.');
  auto filename = GetFileName(img_path);
  std::string save_path = img_path.substr(0, dot_pos + 1) + "txt";
  std::ofstream ofs(save_path);
  if (!ofs.is_open()) {
    std::cout << "ofstream is not opened" << std::endl;
    return;
  }
  ofs << filename << std::endl;
  ofs << bboxes.size() << std::endl;
  for (auto bbox : bboxes) {
    ofs << static_cast<int>(std::round(bbox.cx - bbox.w / 2.0f)) << ' '
	<< static_cast<int>(std::round(bbox.cy - bbox.h / 2.0f)) << ' '
	<< bbox.w << ' '
	<< bbox.h << ' '
	<< bbox.score << std::endl;
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

int main(int argc, char **argv){
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
  std::vector<magicmind::IRTTensor*> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);

  // 6. input tensor memory alloc
  void *mlu_addr_ptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = FLAGS_batch;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
  MM_CHECK(input_tensors[0]->SetData(mlu_addr_ptr));

  // 7. load image
  std::cout << "================== Load Images ====================" << std::endl;
  std::vector<std::string> image_paths = LoadImages(FLAGS_image_dir, FLAGS_batch, FLAGS_image_num, FLAGS_file_list);
  if (image_paths.size() == 0) {
    std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
    return 0;
  }
  size_t image_num = image_paths.size();
  std::cout << "Total images: " << image_num << std::endl;

  std::cout << "Start run..." << std::endl;
  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < image_num; i++) {
    string image_name = GetFileName(image_paths[i]);
    std::cout << "Inference img: " << image_name << "\t\t\t" << i+1 << "/" << image_num << std::endl;
    cv::Mat img = cv::imread(image_paths[i]);
    Mat img_pre = Preprocess(img, input_dim[1], input_dim[2]);
    CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pre.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // 8. compute
    output_tensors.clear();
    MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 9. copy out
    std::vector<std::vector<float>> outputs;
    std::vector<const float *> results;
    for (int j = 0; j < output_tensors.size(); j++) {
      std::vector<float> output(output_tensors[j]->GetSize());
      CNRT_CHECK(cnrtMemcpy((void *)output.data(), output_tensors[j]->GetMutableData(), output_tensors[j]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      outputs.emplace_back(std::move(output));
      results.emplace_back(outputs[j].data());
    }

    auto output_dims = model->GetOutputDimensions();
    std::vector<BBox> bboxes = Postprocess(img, input_dim, results, output_dims);

    char abs_path[PATH_MAX] = {0};
    if (realpath(FLAGS_image_dir.c_str(), abs_path) == NULL) {
      std::cout << "Get real image path in " << FLAGS_image_dir.c_str() << " failed...";
      exit(1);
    }
    std::string image_dir(abs_path);
    std::string image_relative_path = image_paths[i].substr(image_dir.length());
    auto slash_pos = image_relative_path.rfind("/");
    if (slash_pos != std::string::npos) {
      std::string mkdir_cmd = "mkdir -p " + FLAGS_output_dir + "/pred_txts/" + image_relative_path.substr(0, slash_pos);
      system(mkdir_cmd.c_str());
      mkdir_cmd = "mkdir -p " + FLAGS_output_dir + "/images/" + image_relative_path.substr(0, slash_pos);
      system(mkdir_cmd.c_str());
    }

    WritePreds(FLAGS_output_dir + "/pred_txts/" + image_relative_path, bboxes);
    if (FLAGS_save_img) {
      Draw(img, bboxes);
      cv::imwrite(FLAGS_output_dir + "/images/" + image_relative_path, img);
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
  std::cout << "E2E Execution time: " << execution_time.count() << "ms" << std::endl;
  std::cout << "E2E Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps" << std::endl;

  // 10. destroy resource
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
