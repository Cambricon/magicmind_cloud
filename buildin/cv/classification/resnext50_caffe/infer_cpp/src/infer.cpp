#include <mm_runtime.h>
#include <cnrt.h>

#include "../include/pre_process.hpp"
#include "../include/post_process.hpp"
#include "../include/utils.hpp"
using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_int32(image_num, 10, "image number");
DEFINE_string(name_file, "/mm_ws/proj/datasets/imagenet/name.txt", "The label path");
DEFINE_string(label_file, "labels.txt", "labels.txt");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "", "The classification results label output file");
DEFINE_string(result_top1_file, "", "The classification results top1 output file");
DEFINE_string(result_top5_file, "", "The classification results top5 output file");
DEFINE_int32(batch, 1, "The batch size");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 1. cnrt init
    std::cout << "Cnrt init..." << std::endl;
    MluDeviceGuard device_guard(FLAGS_device_id);
    cnrtQueue_t queue;
    // CNRT_CHECK(cnrtQueueCreate, &queue);
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
        std::cout << "You should provide a classification model and check the following:" << std::endl;
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

    // 5. crete input tensor and output tensor and memory alloc
    std::vector<magicmind::IRTTensor*> input_tensors, output_tensors;
    CHECK_MM(context->CreateInputTensors, &input_tensors);
    CHECK_MM(context->CreateOutputTensors, &output_tensors);

    // 6. input tensor memory alloc
    void *mlu_addr_ptr;
    auto input_dim_vec = model->GetInputDimension(0).GetDims();
    auto output_dim_vec = model->GetOutputDimension(0).GetDims();
    if (input_dim_vec[0] == -1) {
      input_dim_vec[0] = FLAGS_batch;
    }
    if (output_dim_vec[0] == -1) {
      output_dim_vec[0] = FLAGS_batch;
    }
    magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
    magicmind::Dims output_dim = magicmind::Dims(output_dim_vec);
    input_tensors[0]->SetDimensions(input_dim);
    CNRT_CHECK(cnrtMalloc(&mlu_addr_ptr, input_tensors[0]->GetSize()));
    MM_CHECK(input_tensors[0]->SetData(mlu_addr_ptr));
    const int classes = output_dim[1];
    const int elem_data_count = output_dim.GetElementCount() / FLAGS_batch;

    // 7. load image
    std::cout << "================== Load Images ====================" << std::endl;
    result *test;
    test = LoadImages(FLAGS_image_dir, FLAGS_batch, FLAGS_image_num, FLAGS_label_file);
    if (test->image_paths.size() == 0) {
       std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
       return 0;
    }
    size_t image_num = test->image_paths.size();
    std::cout << "Total images : " << image_num << std::endl; 

    std::map<int, std::string> name_map = load_name(FLAGS_name_file);
    Record result_label(FLAGS_result_label_file);
    Record result_top1_file(FLAGS_result_top1_file);
    Record result_top5_file(FLAGS_result_top5_file);
    Record result_file(FLAGS_result_file);
    std::cout << "Start run..." << std::endl;
    for (int i = 0 ; i < image_num ; i ++) {
        std::string image_name = test->image_paths[i].substr(test->image_paths[i].find_last_of('/') + 1, 23);
        std::cout << "Inference img : " << test->image_paths[i] << "\t\t\t" << i+1 << "/" << image_num << std::endl;
        cv::Mat img = cv::imread(test->image_paths[i]);
        cv::Mat dst_img = Preprocess(img, input_dim);
        CNRT_CHECK(cnrtMemcpy(input_tensors[0]->GetMutableData(), dst_img.data, input_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));

        // 8. compute
        output_tensors.clear();
        MM_CHECK(context->Enqueue(input_tensors, &output_tensors, queue));
        CNRT_CHECK(cnrtQueueSync(queue));
        
	// 9. copy out
        float *classification_output = nullptr;
        classification_output = (float *)malloc(output_tensors[0]->GetSize());
        CNRT_CHECK(cnrtMemcpy((void *)classification_output, output_tensors[0]->GetMutableData(), elem_data_count * 4 , CNRT_MEM_TRANS_DIR_DEV2HOST));
        
        // get FLAGS_result_label_file
        if (!FLAGS_result_label_file.empty()) {
            result_label.write("[" + std::to_string(i) + "]: " + std::to_string(test->labels[i]), false);
        }
        // get top1 and top5
        std::cout << "get top1 and top5..." << std::endl;
        std::vector<int> top5 = ArgTopK(classification_output, classes, 5);
        // get FLAGS_result_top1_file
        if (!FLAGS_result_top1_file.empty()) {
            result_top1_file.write("[" + std::to_string(i) + "]: " + std::to_string(top5[0]), false);
        }
        // get FLAGS_result_top5_file, FLAGS_result_file
        result_file.write("top5 result in " + image_name + ":", false);
        for (int j = 0 ; j < 5 ; j ++) {
            if (!FLAGS_result_top5_file.empty()) {
                result_top5_file.write("[" + std::to_string(i) + "]: " + std::to_string(top5[j]), false);
            }
            if (!FLAGS_result_file.empty()) {
                result_file.write(std::to_string(j) + " [" + name_map[top5[j]] + "]", false);
            }
        }
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
