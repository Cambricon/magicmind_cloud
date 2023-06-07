#include <mm_runtime.h>
#include <cnrt.h>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

#include "pre_process.h"
#include "post_process.h"
#include "coco_result_saver.h"
#include "utils.h"
#include "model_runner.h"

using namespace magicmind;
using namespace cv;
using namespace std;

#define MINVALUE(A,B) ( A < B ? A : B )

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_int32(image_num, 0, "image number");
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


int main(int argc,char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    TimeCollapse time_openpose_caffe("infer openpose_caffe");

   // create an instance of ModelRunner
    auto cur_model_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
    if (!cur_model_runner->Init(FLAGS_batch_size)) {
      SLOG(ERROR) << "Init cur_model_runner failed.";
      return false;
    }

    // get img_h, img_w
    auto input_dims = cur_model_runner->GetInputDims();
    auto output_dims = cur_model_runner->GetOutputDims();
    const magicmind::Dims input_dim = magicmind::Dims (input_dims[0]);
    int img_h = input_dims[0][1];
    int img_w = input_dims[0][2];

    // load images
    SLOG(INFO) << "================== Load Images ====================";

    std::vector<std::string> image_name_arr= LoadImages(FLAGS_image_dir, FLAGS_image_list);
    if (image_name_arr.size() == 0)
    {
        SLOG(ERROR) << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
        return 0;
    }


    COCOResultSaver coco_saver(FLAGS_output_dir+"/"+GetNetName());
    // batch information
    int batch_counter = 0;
    std::vector<std::string> batch_image_name;
    std::vector<cv::Mat> batch_image;
    std::vector<float> scaling_factors;
    uint8_t *preprocess_data = (uint8_t *)malloc(img_h*img_w*3);

    // allocate host memory for batch preprpcessed data
    auto batch_data = cur_model_runner->GetHostInputData();
    // one batch input data addr offset
    int batch_image_offset = cur_model_runner->GetInputSizes()[0] / FLAGS_batch_size;

    size_t image_num = image_name_arr.size();
    // cambricon-note: if FLAGS_image_num is not set or set to a negative number,the image_num is:image_name_arr.size()
    // if set to a positive num, the image_num is the same as FLAGS_image_num
    if ( FLAGS_image_num > 0 ){
        image_num = MINVALUE(FLAGS_image_num, image_num);
    }
    size_t rem_image_num = image_num % FLAGS_batch_size;
    SLOG(INFO) << "Total images : " << image_num;
    SLOG(INFO) << "Start run...";

    for (int i = 0; i < image_num; i++) {
        std::string image_name = image_name_arr[i].substr(image_name_arr[i].find_last_of('/') + 1, 23);
        SLOG(INFO) << "Inference img : " << image_name << "\t\t\t" << i+1 << "/" << image_num;

        auto image_path = FLAGS_image_dir + "/" + image_name;
        cv::Mat src_img = cv::imread(image_path);
        batch_image.emplace_back(src_img);
        float scaling_factor =  Preprocess(src_img, input_dim, preprocess_data);
        scaling_factors.emplace_back(scaling_factor);

        batch_image_name.push_back(image_name);

        // batching preprocessed data
        memcpy((uint8_t *)(batch_data[0]) + batch_counter * batch_image_offset, preprocess_data, batch_image_offset);

        batch_counter += 1;
        // image_num may not be divisible by FLAGS_batch. 
        // real_batch_size records number of images in every loop, real_batch_size may change in the last loop. 
        size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
        std::cout << "real_batch_size: " << real_batch_size << std::endl;
        if (batch_counter % real_batch_size == 0) {
            // copy in
            cur_model_runner->H2D();
            // compute
            cur_model_runner->Run(FLAGS_batch_size);
            // copy out
            cur_model_runner->D2H();
            // get model's output addr in host

            auto host_output_ptr = cur_model_runner->GetHostOutputData();
            auto infer_res = (float *)host_output_ptr[0];
            auto output_sizes = cur_model_runner->GetOutputSizes();
            auto output_dims = cur_model_runner->GetOutputDims();
            const magicmind::Dims output_dim = Dims(output_dims[0]);
            int elem_count = output_sizes[0] / sizeof(float) / real_batch_size;
            
            // post process
            for (int j = 0; j < real_batch_size; j++) {
            auto cur_pred_res = infer_res + j*elem_count;
            auto res = Postprocess(batch_image[j],cur_pred_res,scaling_factors[j],input_dim,output_dim);
                const Keypoints &keypoints = res.first;
                const PersonInfos &persons = res.second;
            coco_saver.Write(batch_image[j], batch_image_name[j], keypoints, persons);
                if(FLAGS_save_img) {
                    Draw(batch_image[j], keypoints, persons);
                    std::string save_path = FLAGS_output_dir + "/" + batch_image_name[j] + "_rendered.jpg";
                    SLOG(INFO) << "Output features saved in " << save_path;
                    cv::imwrite(save_path, batch_image[j]);
                }//end if FLAGS_save_img

            }// end for j < real_batch_size

            batch_counter = 0;
            batch_image_name.clear();
            batch_image.clear();
            scaling_factors.clear();
        }// end if batch_counter % real_batch_size == 0
    }// end for i < image_num;
    free(preprocess_data);

    // destroy resource
    cur_model_runner->Destroy();
    return 0;
}
