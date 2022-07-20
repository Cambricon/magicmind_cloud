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

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory"); //"./../../../datasets/coco/test";
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(file_list, "file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_string(output_dir, "", "The rendered images output directory");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch, 1, "The batch size");

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
    std::cout << "Total images : " << image_num << std::endl; 

    std::cout << "Start run..." << std::endl; 
    
    for (int i = 0 ; i < image_num ; i ++) {
      string image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
      std::cout << "Inference img : " << image_name << "\t\t\t" << i+1 << "/" << image_num << std::endl;
      cv::Mat img = cv::imread(image_paths[i]);
      cv::Mat img_pro = process_img(img);
      CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), img_pro.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

      // 8. compute
      output_tensors.clear();
      MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
      CNRT_CHECK(cnrtQueueSync(queue));

      // 9. copy out
      vector<vector<float>> results;

      float *data_ptr = nullptr;
      int detection_num;

      CNRT_CHECK(cnrtMemcpy((void *)&detection_num, output_tensors[1]->GetMutableData(), output_tensors[1]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
      data_ptr = (float *)malloc(output_tensors[0]->GetSize());
      CNRT_CHECK(cnrtMemcpy((void *)data_ptr, output_tensors[0]->GetMutableData(), detection_num * 7 * 4, CNRT_MEM_TRANS_DIR_DEV2HOST));
      for(int i = 0 ; i < detection_num ; ++i){
          std::vector<float> result;
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
      free(data_ptr);
      map<int, string> name_map = load_name(FLAGS_label_path);
      post_process(img, results, name_map, image_name, FLAGS_output_dir, FLAGS_save_img);
    }   

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
