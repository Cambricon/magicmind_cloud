#include <gflags/gflags.h>
#include <glog/logging.h>
#include <mm_runtime.h>
#include <cnrt.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>

#include "utils.hpp"
#include "pre_process.hpp"
#include "post_process.hpp"

using namespace std;
using namespace cv;

#define MINVALUE(A,B) ( A < B ? A : B )

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(image_dir, "", "dataset_dir");
DEFINE_string(output_dir,"", "output_dir");
DEFINE_string(label_file, "labels.txt", "labels.txt");
DEFINE_string(name_file, "name.txt", "The label path");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "", "The classification results label output file");
DEFINE_string(result_top1_file, "", "The classification results top1 output file");
DEFINE_string(result_top5_file, "", "The classification results top5 output file");
DEFINE_int32(test_nums, -1, "test_nums");

int main(int argc,char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Cnrt
    cnrtQueue_t queue;
    MluDeviceGuard device_guard(FLAGS_device_id);
    CNRT_CHECK(cnrtQueueCreate(&queue));

   // Model
    auto model = magicmind::CreateIModel();
    MM_CHECK(model->DeserializeFromFile(FLAGS_magicmind_model.c_str()));
    PrintModelInfo(model);

    // Engine
    magicmind::IModel::EngineConfig engine_config;
    engine_config.SetDeviceType("MLU");
    engine_config.SetConstDataInit(true);
    auto engine = model->CreateIEngine(engine_config);
    CHECK_VALID(engine);

    // Context
    auto context = engine->CreateIContext();
    CHECK_VALID(context);
    
    // Input And Output Malloc
    vector<magicmind::IRTTensor *> input_tensors, output_tensors;
    MM_CHECK(context->CreateInputTensors(&input_tensors));
    MM_CHECK(context->CreateOutputTensors(&output_tensors));

    int img_w = 224,img_h=224,img_c=3,batch_size = FLAGS_batch_size,n_classes = 1000;
    
    // malloc
    void *mlu_ptr;
    auto input_dim_vec = model->GetInputDimension(0).GetDims();
    if (input_dim_vec[0] == -1) {
        input_dim_vec[0] = batch_size;
    }

    magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
    input_tensors[0]->SetDimensions(input_dim);
    if (input_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
        CNRT_CHECK(cnrtMalloc(&mlu_ptr, input_tensors[0]->GetSize()));
        MM_CHECK(input_tensors[0]->SetData(mlu_ptr));
    } 

    bool dynamic_output = false;
    if (magicmind::Status::OK() ==
        context->InferOutputShape(input_tensors, output_tensors)) {
        std::cout << "InferOutputShape successed" << std::endl;
        if (output_tensors[0]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
            CNRT_CHECK(cnrtMalloc(&mlu_ptr, output_tensors[0]->GetSize()));
            MM_CHECK(output_tensors[0]->SetData(mlu_ptr));
        } else {
            std::cout << "InferOutputShape failed" << std::endl;
            dynamic_output = true;
        }
    }

    //Set Input data ptr
    uint8_t *input_data_ptr = new uint8_t[batch_size*img_h*img_w*img_c];
    //Set Output data ptr
    float *output_data_ptr = new float[output_tensors[0]->GetSize()/sizeof(output_tensors[0]->GetDataType())];

    // load images
    std::cout << "================== Load Images ====================" << std::endl;
    result *test;
    test = LoadImages(FLAGS_image_dir, batch_size, FLAGS_label_file);
    if (test->image_paths.size() == 0) {
       std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
       return 0;
    }
    size_t image_num = test->image_paths.size();
    std::map<int, std::string> name_map = load_name(FLAGS_name_file);
    // cout << name_map << endl;
    Record result_label(FLAGS_result_label_file);
    Record result_top1_file(FLAGS_result_top1_file);
    Record result_top5_file(FLAGS_result_top5_file);
    Record result_file(FLAGS_result_file);
    vector<string> image_names(batch_size);
    vector<int> image_label(batch_size);
    cout << "Start run..."<< endl;
    auto start_time = std::chrono::steady_clock::now();
    if ( FLAGS_test_nums != -1 ){
        image_num = MINVALUE(FLAGS_test_nums,image_num);
    }
    std::cout << "Total images : " << image_num << std::endl; 
    for ( int i = 0; i <= image_num; i ++ ){
        if ( i != 0 && i % batch_size == 0 ){
            //Memcpy H2D
            CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(),input_data_ptr, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));
            //Queue
            output_tensors.clear();
            MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
            CNRT_CHECK(cnrtQueueSync(queue));
            //Memcpy D2H
            CNRT_CHECK(cnrtMemcpy(  output_data_ptr, output_tensors[0]->GetMutableData(), \
                                    output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
            //post-processs
            for (int _batch = 0; _batch < batch_size; _batch++ ){
                std::string image_name = image_names[_batch].substr(image_names[_batch].find_last_of('/') + 1, 23);
                if (!FLAGS_result_label_file.empty()) {
                    result_label.write("[" + std::to_string(i-batch_size+_batch) + "]: " + std::to_string(image_label[_batch]), false);
                }
                std::vector<int> top5 = ArgTopK(output_data_ptr+_batch*n_classes, n_classes, 5);
                if (!FLAGS_result_top1_file.empty()) {
                    result_top1_file.write("[" + std::to_string(i-batch_size+_batch) + "]: " + std::to_string(top5[0]), false);
                }
                result_file.write("top5 result in " + image_name + ":", false);
                for (int j = 0 ; j < 5 ; j ++) {
                    if (!FLAGS_result_top5_file.empty()) {
                        result_top5_file.write("[" + std::to_string(i-batch_size+_batch) + "]: " + std::to_string(top5[j]), false);
                    }
                    if (!FLAGS_result_file.empty()) {
                        result_file.write(std::to_string(j) + " [" + name_map[top5[j]] + "]", false);
                    }
                    if ( i == image_num && _batch == ( batch_size - image_num % batch_size - 1 ))
                        break;
                }
            }
        }
        if (i == image_num) continue;
        //pre-process
        cv::Mat src_img = cv::imread(test->image_paths[i]);
        if ( src_img.data != NULL ){
            image_names[ i % batch_size ] = test->image_paths[i];
            image_label[ i % batch_size ] = test->labels[i];
            cv::Mat dst_img(img_h,img_w,CV_8UC3);
            Preprocess(src_img , img_h , img_w , dst_img);
            memcpy(input_data_ptr + ( i % batch_size ) * img_c * img_h * img_w , dst_img.data , img_c * img_h * img_w * magicmind::DataTypeSize(input_tensors[0]->GetDataType()));
        }
    }
    //end free resources
    CNRT_CHECK(cnrtQueueDestroy(queue));
    delete [] input_data_ptr;
    delete [] output_data_ptr;

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
    cout << "All images processed..." << endl;
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
    cout << "Execution time: " << execution_time.count()/1000.0 << "s" << endl;
    cout << "Throughput(1000 / execution time * image number): " << 1000 / execution_time.count() * image_num << "fps"<<endl;
    cout << "Finished!" << endl;
    return 0;
}
