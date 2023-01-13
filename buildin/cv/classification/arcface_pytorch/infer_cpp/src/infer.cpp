#include <mm_runtime.h>
#include <cnrt.h>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pre_process.h"
#include "utils.h"

using namespace magicmind;
using namespace cv;

DEFINE_string(magicmind_model, "", "the magicmind model path");
DEFINE_string(image_dir, "", "predict image file path");
DEFINE_string(image_list, "", "predict image list");
DEFINE_string(output_dir, "", "output path");
DEFINE_int32(image_num, 1000, "image num");
DEFINE_bool(save_img, false, "save img or not. default: false");

int main(int argc, char **argv)
{

    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;
    // 1. cnrt init
    LOG(INFO) << "Cnrt init...";
    uint8_t device_id = 0;
    MluDeviceGuard device_guard(device_id);
    cnrtQueue_t queue;
    CHECK_CNRT(cnrtQueueCreate, &queue);

    // 2. create model
    LOG(INFO) << "Load model...";
    IModel *model = CreateIModel();
    model->DeserializeFromFile(FLAGS_magicmind_model.c_str());
    PrintModelInfo(model);
    // Check if current program can deal with this model
    CHECK_EQ(CheckModel(model), true)
      << "Can not deal with this model.\n"
         "You should check your model with the "
         "following things:\n"
         "1. Make sure the data type of input is UINT8.\n"
         "2. Make sure the input data is in NHWC order.\n"
         "3. Make sure the data type of output is FLOAT.";

    // 3.crete engine
    LOG(INFO) << "Create engine...";
    auto engine = model->CreateIEngine();
    CHECK_PTR(engine);
    magicmind::IModel::EngineConfig engine_config;
    engine_config.SetDeviceType("MLU");
    engine_config.SetConstDataInit(true);

    // 4.create context
    auto context = engine->CreateIContext();
    CHECK_PTR(context);

    // 5.crete input tensor and output tensor and memory alloc
    std::vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    CHECK_MM(context->CreateInputTensors, &input_tensors);
    CHECK_MM(context->CreateOutputTensors, &output_tensors);
    CHECK_STATUS(context->InferOutputShape(input_tensors, output_tensors));
    LOG(INFO) << "input_tensors:"<<input_tensors[0]->GetSize();

    auto input_dim = model->GetInputDimension(0);
    auto output_dim = model->GetOutputDimension(0);
    auto batch_size = input_dim[0];
    // 6.input tensor memory alloc
    for (auto tensor : input_tensors)
    {
        void *mlu_addr_ptr;
        CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
        CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
    }

    //   output tensor memory alloc
    for (auto tensor : output_tensors)
    {
        void *mlu_addr_ptr;
        CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, tensor->GetSize()));
        CHECK_STATUS(tensor->SetData(mlu_addr_ptr));
    }

    // 7. load image and label
    LOG(INFO) << "================== Load Images ====================";
    std::vector<std::string> image_paths = LoadImages(FLAGS_image_dir, FLAGS_image_list, FLAGS_image_num, input_dim[0]);
    if (image_paths.size() == 0)
    {
        LOG(INFO) << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
        return 0;
    }
    size_t image_num = image_paths.size();
    LOG(INFO) << "Total images : " << image_num << std::endl;
    LOG(INFO) << "Start run..." << std::endl;
    std::vector<std::string> image_names;
    std::vector<std::string> faceness_scores;
    for (int i = 0; i < image_num; i += batch_size)
    {
        for (int j = 0 ; j < batch_size ; j ++) {
            auto line = image_paths[i + j];
            std::vector<std::string> res = SplitString(line);
            std::string path = FLAGS_image_dir + "/" + res[0];
            std::string image_name = GetFileName(path);
            image_names.emplace_back(image_name);
            int end_pos = res[0].find('.');
            std::string id = res[0].substr(0, end_pos);
            std::string faceness_score = res[11];
            faceness_scores.emplace_back(faceness_score);
            std::vector<std::string> landmarks;
            landmarks.insert(landmarks.end(), res.begin() + 1, res.end() - 1);
            if (!check_file_exist(path))
            {
                LOG(INFO) << "image file " + path + " not found.\n";
                exit(1);
            }
            LOG_EVERY_N(INFO,100) << "Inference img: " << path << "\t\t\t" << i << "/" << image_num << std::endl;
            cv::Mat img = cv::imread(path);
            if (img.empty())
            {
                LOG(INFO) << "Failed to open image file " + image_paths[i];
                exit(1);
            }
            //LOG(INFO) << "Inference img: " << image_name << "\t\t\t" << i << "/" << image_num << std::endl;
            cv::Mat input_img =  Preprocess(img, input_dim, landmarks);
            // 8. copy in
            CNRT_CHECK(cnrtMemcpy((uint8_t *)(input_tensors[0]->GetMutableData()) + j * (input_tensors[0]->GetSize() / batch_size) , input_img.data, input_tensors[0]->GetSize() / batch_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        }
        // 9. compute
        CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
        CNRT_CHECK(cnrtQueueSync(queue));

        // 10. copy out
        void *output_cpu_ptrs = (void *)malloc(output_tensors[0]->GetSize());
        CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
        std::vector<float> output_data((float *)output_cpu_ptrs, (float *)output_cpu_ptrs + (output_tensors[0]->GetSize() / sizeof(float)));

        if (FLAGS_save_img)
        {
            for(int j = 0 ; j < batch_size ; j++ ) {
                std::string save_path = FLAGS_output_dir + "/" + image_names[j] + ".feature";
                LOG_EVERY_N(INFO,1000) << "Output features saved in " << save_path;
                std::ofstream ofs(save_path);
                LOG_IF(FATAL, !ofs.is_open()) << "Create file [" << save_path << "] failed.";
                for (int k = 0; k < output_dim[1]; k++)
                {
                    ofs << ((float *)output_cpu_ptrs)[k + j * output_dim[1]] << " ";
                }
                ofs << faceness_scores[j] <<std::endl;
                ofs.close();
            }
        }
        image_names.clear();
        faceness_scores.clear();
    }
    // 8. destroy resource
    for (auto tensor : input_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    for (auto tensor : output_tensors)
    {
        cnrtFree(tensor->GetMutableData());
        tensor->Destroy();
    }
    context->Destroy();
    engine->Destroy();
    model->Destroy();
    return 0;
}
