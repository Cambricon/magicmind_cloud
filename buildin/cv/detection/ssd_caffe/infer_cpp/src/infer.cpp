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

#include "../include/pre_process.hpp"
#include "../include/post_process.hpp"
#include "../include/utils.hpp"

#define max_box 100

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch, 1, "The batch size");

static std::vector<std::string> glabels = {
    "__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
};

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
    vector<magicmind::IRTTensor*> input_tensors, output_tensors;
    CHECK_MM(context->CreateInputTensors, &input_tensors);
    CHECK_MM(context->CreateOutputTensors, &output_tensors);

    // 5. memory alloc
    // input tensor mlu ptrs
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
    // output tensor mlu ptrs
    bool dynamic_output = false;
    if (magicmind::Status::OK() == context->InferOutputShape(input_tensors, output_tensors)) {
      std::cout << "InferOutputShape successed" << std::endl;
      if (output_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
        CNRT_CHECK(cnrtMalloc(&ptr, output_tensors[0]->GetSize()));
        MM_CHECK(output_tensors[0]->SetData(ptr));
      } 
    } else {
        std::cout << "InferOutputShape failed" << std::endl;
        dynamic_output = true;
    }

    //output_tensor cpu ptrs
    float *data_ptr = nullptr;
    data_ptr = new float[max_box * 7];
    int detection_num;

    // 6. load image
    std::cout << "================== Load Images ====================" << std::endl;
    std::vector<std::string> image_paths = LoadImages(FLAGS_image_dir, FLAGS_batch);
    if (image_paths.size() == 0) {
       std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
       return 0;
    }
    size_t image_num = image_paths.size();
    std::cout << "Total images : " << image_num << std::endl; 
    std::cout << "Start run..." << std::endl; 
    for (int i = 0 ; i < image_num ; i ++) {
      vector<vector<float>> results;
      auto begin = image_paths[i].find_last_of('/');
      auto end = image_paths[i].find_last_of('.');
      auto len = end - begin - 1;
      string image_name = image_paths[i].substr(begin + 1, len);
      std::cout << "Inference img : " << image_name << "\t\t\t" << i+1 << "/" << image_num << std::endl;
      Mat img = imread(image_paths[i]);
      Mat img_pro = process_img(img);
      CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

      // 8. compute
      // output_tensors.clear();
      MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
      CNRT_CHECK(cnrtQueueSync(queue));

      // 9. copy out
      detection_num = output_tensors[0]->GetSize() / 7 / 4;
      CNRT_CHECK(cnrtMemcpy((void *)data_ptr, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      for(int i = 0 ; i < detection_num ; ++i){
          std::vector<float> result;
          float batch_id = (int)(*(data_ptr+7*i));
          float class_idx = *(data_ptr+7*i+1);
          float score = *(data_ptr+7*i+2);
          float xmin = *(data_ptr+7*i+3);
          float ymin = *(data_ptr+7*i+4);
          float xmax = *(data_ptr+7*i+5);
          float ymax = *(data_ptr+7*i+6);

          result.push_back(class_idx);
          result.push_back(score);
          result.push_back(xmin);
          result.push_back(ymin);
          result.push_back(xmax);
          result.push_back(ymax);
          results.push_back(result);
      }
      std::vector<std::string> voc_preds_files;
      for (int t = 0 ; t < glabels.size() ; t ++) {
          std::string voc_preds_file = FLAGS_output_dir + "/" + "voc_preds/comp3_det_test_" + glabels[t] + ".txt";
          voc_preds_files.push_back(voc_preds_file);
      }
      post_process(img, results, glabels, voc_preds_files, image_name, FLAGS_output_dir, FLAGS_save_img);
    }

    // 10. destroy resource
    // destroy must do strictly as follow
    // destroy tensor/address first
    delete[] data_ptr;
    for (auto tensor : input_tensors) {
      if (tensor->GetMemoryLocation() == magicmind::TensorLocation::kMLU){
        cnrtFree(tensor->GetMutableData());
      }
      tensor->Destroy();
    }
    for (auto tensor : output_tensors) {
      if (!dynamic_output){
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

