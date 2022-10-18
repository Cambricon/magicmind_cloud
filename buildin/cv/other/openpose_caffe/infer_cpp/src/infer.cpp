#include <mm_runtime.h>
#include <cnrt.h>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pre_process.h"
#include "post_process.h"
#include "coco_result_saver.h"
#include "utils.h"

using namespace magicmind;
using namespace cv;

DEFINE_string(magicmind_model, "", "the magicmind model path");
DEFINE_string(image_dir, "", "predict image file path");
DEFINE_string(image_list, "", "predict image list");
DEFINE_string(output_dir, "", "output path");
DEFINE_bool(save_img, false, "save img or not. default: false");
DEFINE_double(render_threshold, 0.05, "The people with score greater thaan render_threshold will be rendered.");

void Draw(cv::Mat img, const Keypoints &keypoints, const PersonInfos &persons) {
  const NetParams &net_params = GetNetParams();
  const auto &colors = net_params.colors;
  int ncolors = static_cast<int>(colors.size());
  for (const auto &person : persons) {
    // draw lines
    for (size_t i = 0; i < net_params.render_body_part_pairs.size(); ++i) {
      const cv::Point &body_pair = net_params.render_body_part_pairs[i];
      const cv::Point &idxa = person.keypoint_idxs[body_pair.x];
      const cv::Point &idxb = person.keypoint_idxs[body_pair.y];
      if (idxa != kInvalidPos && idxb != kInvalidPos) {
        const KeypointInfo &a = keypoints[idxa.x][idxa.y];
        const KeypointInfo &b = keypoints[idxb.x][idxb.y];
        if (a.score > FLAGS_render_threshold && b.score > FLAGS_render_threshold)
          cv::line(img, RoundPoint(a.pos, img), RoundPoint(b.pos, img), colors[i % ncolors], 2, 8, 0);
      }
    }  // for body pairs
    // draw keypoints
    for (const auto &idx : person.keypoint_idxs) {
      if (idx != kInvalidPos) {
        const auto &keypoint = keypoints[idx.x][idx.y];
        if (keypoint.score > FLAGS_render_threshold)
          cv::circle(img, RoundPoint(keypoint.pos, img), 2, cv::Scalar(0, 0, 255), 1, 8, 0);
      }  // if invalid pos
    }  // for keypoints
  }  // for persons
}


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
    std::vector<std::string> image_paths = LoadImages(FLAGS_image_dir, FLAGS_image_list, input_dim[0]);
    if (image_paths.size() == 0)
    {
        LOG(INFO) << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
        return 0;
    }
    size_t image_num = image_paths.size();
    LOG(INFO) << "Total images : " << image_num << std::endl;
    LOG(INFO) << "Start run..." << std::endl;
    COCOResultSaver coco_saver(FLAGS_output_dir+"/"+GetNetName());
    std::vector<std::string> image_names;
    std::vector<cv::Mat> images;
    std::vector<float> scaling_factors;
    uint8_t *preprocess_data = (uint8_t *)malloc(input_dim[1]*input_dim[2]*3);
    for (int i = 0; i < image_num; i += batch_size)
    {
        for (int j = 0 ; j < batch_size ; j ++) {
            auto line = image_paths[i + j];
            std::vector<std::string> res = SplitString(line);
            std::string path = FLAGS_image_dir + "/" + res[0];
            std::string image_name = GetFileName(path);
            image_names.emplace_back(image_name);
            if (!check_file_exist(path))
            {
                LOG(INFO) << "image file " + path + " not found.\n";
                exit(1);
            }
            LOG_EVERY_N(INFO,100) << "Inference img: " << path << "\t\t\t" << i << "/" << image_num << std::endl;
            cv::Mat img = cv::imread(path);
	    images.emplace_back(img);
            if (img.empty())
            {
                LOG(INFO) << "Failed to open image file " + image_paths[i];
                exit(1);
            }
            //LOG(INFO) << "Inference img: " << image_name << "\t\t\t" << i << "/" << image_num << std::endl;
            float scaling_factor =  Preprocess(img, input_dim, preprocess_data);
	    scaling_factors.emplace_back(scaling_factor);
            // 8. copy in
            CNRT_CHECK(cnrtMemcpy((uint8_t *)(input_tensors[0]->GetMutableData()) + j * (input_tensors[0]->GetSize() / batch_size) , preprocess_data, input_tensors[0]->GetSize() / batch_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
        }
        // 9. compute
        CHECK_STATUS(context->Enqueue(input_tensors, output_tensors, queue));
        CNRT_CHECK(cnrtQueueSync(queue));

        // 10. copy out
        void *output_cpu_ptrs = (void *)malloc(output_tensors[0]->GetSize());
        CNRT_CHECK(cnrtMemcpy(output_cpu_ptrs, output_tensors[0]->GetMutableData(), output_tensors[0]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
	int elem_count = output_tensors[0]->GetSize() / sizeof(float) / batch_size;
        if (!FLAGS_output_dir.empty()) {
            for(int j = 0 ; j < batch_size ; j++ ) {
		auto res = Postprocess(images[j], (float *)output_cpu_ptrs + j * elem_count,
                               scaling_factors[j], input_dim, output_dim);
                const Keypoints &keypoints = res.first;
                const PersonInfos &persons = res.second;
		coco_saver.Write(images[j], image_names[j], keypoints, persons);
                if(FLAGS_save_img) {
                    Draw(images[j], keypoints, persons);
                    std::string save_path = FLAGS_output_dir + "/" + image_names[j] + "_rendered.jpg";
                    LOG_EVERY_N(INFO,1000) << "Output features saved in " << save_path;
                    cv::imwrite(save_path, images[j]);
		}
            }
        }
        image_names.clear();
	images.clear();
	scaling_factors.clear();
    }
    free(preprocess_data);
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
