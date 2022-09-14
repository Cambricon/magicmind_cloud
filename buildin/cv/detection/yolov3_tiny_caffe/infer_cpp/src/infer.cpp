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

DEFINE_int32(save_img ,0, "save_img");
DEFINE_int32(save_pred,1, "save_pred");
DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(image_dir, "", "dataset_dir");
DEFINE_int32(pad_value, 128, "The pad value for image preprocessing");
DEFINE_string(output_img_dir,"", "output_dir");
DEFINE_string(output_pred_dir,"", "output_dir");
DEFINE_double(confidence_thresholds, 0.001, "Confidence thresholds");
DEFINE_string(label_path, "", "The label path");
DEFINE_string(save_imgname_dir, "", "save_imgname_dir");


const int img_h = 416;
const int img_w = 416;
const int img_c = 3;

int main(int argc,char **argv)
{
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Cnrt
    cnrtQueue_t queue;
    MluDeviceGuard device_guard(FLAGS_device_id);
    CNRT_CHECK(cnrtQueueCreate(&queue)); //mm0.12 cnrtCreateQueue  

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

    // malloc
    void *mlu_input_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_input_addr_ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(mlu_input_addr_ptr));

    void *mlu_output_1_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_output_1_addr_ptr, output_tensors[0]->GetSize()));
    MM_CHECK(output_tensors[0]->SetData(mlu_output_1_addr_ptr));
    
    void *mlu_output_2_addr_ptr;
    CNRT_CHECK(cnrtMalloc(&mlu_output_2_addr_ptr, output_tensors[1]->GetSize()));
    MM_CHECK(output_tensors[1]->SetData(mlu_output_2_addr_ptr));

    //Set Input data ptr
    int batch_size = FLAGS_batch_size;
    uint8_t *input_data_ptr = new uint8_t[batch_size*img_h*img_w*img_c];
    // cout << batch_size*img_h*img_w*img_c << endl;

    //Set Output data ptr
    float *output_0_data_ptr = new float[output_tensors[0]->GetSize()/sizeof(output_tensors[0]->GetDataType())];
    int32_t* output_1_data_ptr = new int32_t[output_tensors[1]->GetSize()/sizeof(output_tensors[1]->GetDataType())];

    // load images
    cout << "Load images..." << endl;
    std::vector<cv::String> image_paths = LoadImages(batch_size,FLAGS_image_dir); //传入输入bs 防止模型可变bs 内部已处理好为bs的整数倍
    if (image_paths.size() == 0) {
        cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
        return 0;
    }

    // load labels 
    auto labels = LoadLabels(FLAGS_label_path);
    size_t image_num = image_paths.size();
    cout << "Total images : " << image_num <<  endl;

    string save_pred_dir = FLAGS_output_pred_dir;
    string save_img_dir = FLAGS_output_img_dir;
    bool save_img = FLAGS_save_img  > 0 ? true:false;
    bool save_pred = FLAGS_save_pred > 0 ? true:false;

    //INPUT:uint8 Output[0] FLOAT Output[1] INT32
    vector<string> image_name(batch_size);
    cv::Mat dst_img(img_h,img_w,CV_8UC3);
    vector<float> scaling_factors(batch_size);
    vector<cv::Mat> src_imgs(batch_size);
    cout << "Start run..."<< endl;
    string save_filelist = FLAGS_save_imgname_dir + '/'  + "image_name.txt";
    std::ofstream filelist(save_filelist);
    auto start_time = std::chrono::steady_clock::now();
    image_num = (int)(image_num/5);//小批量用于CI-TEST 正式使用时为image_num
    for ( int i = 0; i < image_num; i ++ ){
        if ( i != 0 && i % batch_size == 0 ){
            //Memcpy H2D
            CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(),input_data_ptr, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));
            //Queue
            output_tensors.clear();
            MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
            CNRT_CHECK(cnrtQueueSync(queue));
            //Memcpy D2H
            CNRT_CHECK(cnrtMemcpy(  output_0_data_ptr, output_tensors[0]->GetMutableData(), \
                                    output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
            CNRT_CHECK(cnrtMemcpy(  output_1_data_ptr, output_tensors[1]->GetMutableData(), \
                                    output_tensors[1]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
            //postprocess
            int32_t output_0_size = output_tensors[0]->GetSize();
            for (int num_ = 0 ; num_ < batch_size; num_ ++ ){
                //get bbox
                std::vector<BBox> bboxes =  Yolov3GetBBox(  src_imgs[ num_ ], scaling_factors[ num_ ], img_h, img_w,\
                                            &output_0_data_ptr[num_ * output_0_size / (batch_size*sizeof(float))], \
                                            output_1_data_ptr[num_], FLAGS_confidence_thresholds);
                //post-process
                filelist << image_name[num_] << "\n";
                saveImgAndPreds(save_img,save_pred,src_imgs, image_name,save_img_dir,save_pred_dir,labels,num_,bboxes);
            }
        }

        cv::Mat src_img = cv::imread(image_paths[i]);
        if ( src_img.data != NULL ){
            image_name[ i % batch_size ] = image_paths[i];
            //pre-process
            scaling_factors[ i % batch_size ] = Preprocess(src_img , img_h , img_w , dst_img , FLAGS_pad_value);
            src_imgs[ i % batch_size ] = src_img;
            memcpy(input_data_ptr + ( i % batch_size ) * img_c * img_h * img_w , dst_img.data , img_c * img_h * img_w * magicmind::DataTypeSize(input_tensors[0]->GetDataType())); //*sizeof(uint8_t) 传输多少个字节的数据
        }
    }
    filelist.close();
    //end free resources
    CNRT_CHECK(cnrtQueueDestroy(queue));
    delete [] input_data_ptr;
    delete [] output_0_data_ptr;
    delete [] output_1_data_ptr;
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