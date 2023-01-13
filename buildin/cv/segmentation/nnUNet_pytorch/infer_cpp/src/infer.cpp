#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <cstring>
#include <chrono>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// #include "../include/pre_process.hpp"
#include "../include/utils.hpp"

using namespace magicmind;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(data_folder, "", "The data folder");
DEFINE_string(output_folder, "", "The rendered images output directory");
DEFINE_int32(batch, 4, "The batch size");
void softmax_pixel(float *ptr, int h, int w, int c)
{
  for (int h_id = 0; h_id < h; h_id++)
  {
    for (int w_id = 0; w_id < w; w_id++)
    {
      float sum = 0.0;
      for (int c_id = 0; c_id < c; c_id++)
      {
        float value = ptr[h_id * w * c + w_id * c + c_id];
        ptr[h_id * w * c + w_id * c + c_id] = exp(value);
        sum += ptr[h_id * w * c + w_id * c + c_id];
      }
      for (int c_id = 0; c_id < c; c_id++)
      {
        ptr[h_id * w * c + w_id * c + c_id] /= sum;
      }
    }
  }
}

static std::vector<cv::String> GetFileList(std::string rex_string, std::string dir) {
  char abs_path[PATH_MAX];
  if (dir.empty()) {
    std::cout << "dir: " + dir + "is empty" << std::endl;
    abort();
  }
  if (!realpath(dir.c_str(), abs_path)) { std::cout << "Get " + dir + " failed."<< std::endl;}
  std::string glob_path = std::string(abs_path);
  std::vector<cv::String> file_paths;
  cv::glob(glob_path + rex_string, file_paths, false);
  return file_paths;
}


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 1. cnrt init
  std::cout << "Cnrt init..." << std::endl;
  MluDeviceGuard device_guard(FLAGS_device_id);
  cnrtQueue_t queue;
  CHECK_CNRT(cnrtQueueCreate, &queue);

  // 2. create model
  std::cout << "Load model..." << std::endl;
  IModel *model = CreateIModel();
  CHECK_PTR(model);
  model->DeserializeFromFile(FLAGS_magicmind_model.c_str());
  PrintModelInfo(model);

  // 3.crete engine
  std::cout << "Create engine..." << std::endl;
  auto engine = model->CreateIEngine();
  CHECK_PTR(engine);

  // 4.create context
  std::cout << "Create context..." << std::endl;
  magicmind::IModel::EngineConfig engine_config;
  engine_config.SetDeviceType("MLU");
  engine_config.SetConstDataInit(true);
  auto context = engine->CreateIContext();
  CHECK_PTR(context);

  // 5.crete input tensor and output tensor and memory alloc
  std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
  CHECK_MM(context->CreateInputTensors, &input_tensors);
  CHECK_MM(context->CreateOutputTensors, &output_tensors);
  // MM_CHECK(context->InferOutputShape(input_tensors, output_tensors));

  // 6.input tensor memory alloc
  void *ptr = nullptr;
  auto input_dim_vec = model->GetInputDimension(0).GetDims();
  if (input_dim_vec[0] == -1) {
    input_dim_vec[0] = FLAGS_batch;
  }
  magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
  input_tensors[0]->SetDimensions(input_dim);
  if (input_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
    CNRT_CHECK(cnrtMalloc(&ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(ptr));
  }

  //  output tensor memory alloc in device
  auto output_dim_vec = model->GetOutputDimension(0).GetDims();
  if (output_dim_vec[0] == -1) {
    output_dim_vec[0] = FLAGS_batch;
  }
  magicmind::Dims output_dim = magicmind::Dims(output_dim_vec);
  bool dynamic_output = false;
  if (magicmind::Status::OK() == context->InferOutputShape(input_tensors, output_tensors)) {
    std::cout << "InferOutputShape successed" << std::endl;
    for (size_t output_id = 0; output_id < model->GetOutputNum(); ++output_id) {
      if (output_tensors[output_id]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
        CNRT_CHECK(cnrtMalloc(&ptr, output_tensors[output_id]->GetSize()));
        MM_CHECK(output_tensors[output_id]->SetData(ptr));
      }
    }
  } else {
      std::cout << "InferOutputShape failed" << std::endl;
      dynamic_output = true;
  }

  // output tensor malloc in host
  float *output_cpu_ptrs = new float[output_tensors[0]->GetSize()/sizeof(output_tensors[0]->GetDataType())];

  // 7. load image
  std::cout << "================== Load Images ====================" << std::endl;
  std::vector<cv::String> file_lists = GetFileList("/*data", FLAGS_data_folder);
  std::cout << "File list size: " << file_lists.size() << std::endl;
  std::cout << "Start run..." << std::endl;
  std::vector<std::vector<int>> data_shapes;
  for (int file_index = 0; file_index < file_lists.size(); ++file_index) {
    auto data_shape_path = file_lists[file_index] + "_shape_info";
    std::vector<int> data_shape = ReadBinFile<int>(data_shape_path);
    data_shapes.emplace_back(std::move(data_shape));
  }
  for (int i = 0; i < file_lists.size(); i++) {
    std::vector<int> data_shape = data_shapes[i];
    std::string data_path = file_lists[i];
    std::vector<float> data = ReadBinFile<float>(data_path);
    float *data_ptr = data.data();
    int image_elemt_size = (int)(data_shape[2] * data_shape[3]);
    for (int index = 0; index < data_shape[1]; index++) {
      auto img_data = data_ptr + index * image_elemt_size;
      cv::Mat img(data_shape[2], data_shape[3], CV_32FC1, img_data);
      cv::Mat padded_img;
      int h = std::max((int64_t)data_shape[2], input_dim[1]);
      int w = std::max((int64_t)data_shape[3], input_dim[2]);
      int pad_h_before = (int)((h - data_shape[2]) / 2);
      int pad_h_after = (int)(h - data_shape[2] - pad_h_before);
      int pad_w_before = (int)((w - data_shape[3]) / 2);
      int pad_w_after = (int)(w - data_shape[3] - pad_w_before);
      cv::copyMakeBorder(img, padded_img, pad_h_before, pad_h_after,
                        pad_w_before, pad_w_after, cv::BORDER_CONSTANT,
                        cv::Scalar(0));
      for (int mirror_index = 0; mirror_index < FLAGS_batch; mirror_index++)
      {
        cv::Mat rgb(input_dim[1], input_dim[2], CV_32FC1);
        if (mirror_index % FLAGS_batch== 1)
        {
          cv::flip(padded_img, rgb, 0);
        }
        else if (mirror_index % FLAGS_batch == 2)
        {
          cv::flip(padded_img, rgb, 1);
        }
        else if (mirror_index % FLAGS_batch == 3)
        {
          cv::flip(padded_img, rgb, -1);
        }
        else
        {
          padded_img.copyTo(rgb);
        }

        // 8. copy in
        CNRT_CHECK(cnrtMemcpy(((float *)input_tensors[0]->GetMutableData()) + mirror_index * input_tensors[0]->GetSize() / FLAGS_batch / sizeof(float),
                              rgb.data, input_tensors[0]->GetSize() / FLAGS_batch, CNRT_MEM_TRANS_DIR_HOST2DEV));
      }

      //  9. compute
      MM_CHECK(context->Enqueue(input_tensors, output_tensors, queue));
      CNRT_CHECK(cnrtQueueSync(queue));

      std::vector<float *> ptrs;
      std::vector<cv::Mat> imgs;
      int elem_data_count = output_dim.GetElementCount() / FLAGS_batch;
      for (int batch_id = 0; batch_id < FLAGS_batch; batch_id++)
      {
        ptrs.emplace_back(((float *)output_cpu_ptrs) + batch_id * elem_data_count);
        cv::Mat img(output_dim[1], output_dim[2], CV_32FC2, ptrs[batch_id]);
        imgs.emplace_back(std::move(img));
      }

      // 10. copy out
      CNRT_CHECK(cnrtMemcpy((void *)output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      for (int batch_id = 0; batch_id < FLAGS_batch; batch_id++) {
        softmax_pixel(ptrs[batch_id], output_dim[1], output_dim[2], output_dim[3]);
        cv::addWeighted(imgs[0], 0.25, imgs[batch_id], 0.25, 0.0, imgs[0]);
      }
      cv::Mat crop_img = imgs[0](cv::Rect(pad_w_before, pad_h_before, data_shape[3], data_shape[2])).clone();
      float *crop_data = (float *)crop_img.data;
      int count = data_shape[2] * data_shape[3] * 2;
      int pos = data_path.find_last_of('/');
      std::string data_file_name(data_path.substr(pos + 1));
      if (!FLAGS_output_folder.empty()) {
        // seg_output
        // compute argmax
        std::vector<uint8_t> argmax_result;
        for (int k = 0; k < count / 2 ; k++) {
          if (crop_data[2 * k] > crop_data[2 * k + 1]) {

            argmax_result.push_back((uint8_t)0);
          } else {
            argmax_result.push_back((uint8_t)255);
          }
        }
        cv::Mat seg_img(data_shape[2], data_shape[3], CV_8UC1, argmax_result.data());
        std::string seg_output_path = FLAGS_output_folder + "/" +
                                data_file_name + "_output_" +
                                std::to_string(index) + "_seg.jpg";
        cv::imwrite(seg_output_path, seg_img);
        // softmax_output
        std::string softmax_output_path = FLAGS_output_folder + "/" +
                                data_file_name + "_output_" +
                                std::to_string(index);
        WriteBinFile(softmax_output_path, count, crop_data);
      }
    }
  }

  // 8. destroy resource
  // destroy must do strictly as follow
  // destroy tensor/address first
  delete[] output_cpu_ptrs;
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
