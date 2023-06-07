#include <gflags/gflags.h>
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
#include "model_runner.h"

using namespace std;
using namespace cv;

#define MINVALUE(A, B) (A < B ? A : B)

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_int32(clip_step, -1, "The distance between each video clip, -1 means equal to sampling_rate * clip_length");
DEFINE_int32(sampling_rate, 2, "The sampling rate for video clips");
DEFINE_string(video_list, "", "Video list file path.");
DEFINE_int32(batch_size, 8, "batch_size");
DEFINE_string(image_dir, "", "dataset_dir");
DEFINE_string(name_file, "name.txt", "The label path");
DEFINE_string(result_file, "", "The classification results output file");
DEFINE_string(result_label_file, "", "The classification results label output file");
DEFINE_string(result_top1_file, "", "The classification results top1 output file");
DEFINE_string(result_top5_file, "", "The classification results top5 output file");
DEFINE_int32(image_num, 0, "image number");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  TimeCollapse time_c3d_caffe("infer c3d_caffe");
  // create an instance of ModelRunner
  auto cur_model_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
  if (!cur_model_runner->Init(FLAGS_batch_size)) {
    SLOG(ERROR) << "Init cur_model_runner failed.";
    return false;
  }

  // set basic vars.
  int img_w = 112, img_h = 112, img_c = 3, clip_len = 8, n_classes = 101;
  int clip_steps = (FLAGS_sampling_rate - 1)*clip_len;
  int batch_size = FLAGS_batch_size;
  int img_chw = img_w*img_h*img_c;

  // load images
  SLOG(INFO) << "================== Loading Videos ... ====================";
  result *test;
  test = loadVideosAndLabels(FLAGS_video_list,FLAGS_name_file);
  int total_videos = test->video_paths.size();
  int total_clips_id = 0;

  if (total_videos == 0) {
      SLOG(ERROR) << "No videos in video list[" << FLAGS_video_list << "].";
    return 1;
  }
  if (FLAGS_image_num > 0){
      total_videos = min(FLAGS_image_num,total_videos);
  }
  SLOG(INFO) << "All Videos Num: " << total_videos << "\n";

  // batch information
  Record result_label(FLAGS_result_label_file);
  Record result_top1_file(FLAGS_result_top1_file);
  Record result_top5_file(FLAGS_result_top5_file);
  Record result_file(FLAGS_result_file);

  cv::VideoCapture capture;
  cv::Mat frame;

  // allocate host memory for batch preprpcessed data
  auto batch_data = cur_model_runner->GetHostInputData();

  SLOG(INFO) << "Start run...";
  for(int i=0;i<total_videos;++i){
    // start to traverse all the videos.
    SLOG(INFO)<< i+1 <<"/"<<total_videos<<"\n";
    cv::VideoCapture capture;
    std::string real_path = FLAGS_image_dir + "/" + test->video_paths[i];
    if(!capture.open(real_path)){
        SLOG(ERROR)<<"Faild open "<<real_path<<"\n";
        break;
    }// end if capture open
    bool video_end = false;// the current video reaches its end.
    int clip_id = 0;// the valid part of the current video.

    // proc one video
    while(!video_end){
        //pre process
        int per_vaild_batch = 0;
        int pass_frames = 0;

        int offset_gap =0;
        int prev_offset=0;

        for(int bs_index =0;bs_index<batch_size;++bs_index){
            //积累batch
            int start =0;
            while(start<(pass_frames + clip_len)){
                if(capture.read(frame)){
                    if(start>=pass_frames){
                        cv::Mat ret = PreprocessImage(frame);

                        auto batch_off_set = (bs_index*clip_len+(start-pass_frames))*img_chw;
                        auto offset_gap = batch_off_set -prev_offset;
                        prev_offset = batch_off_set;
                        auto input_data_type = cur_model_runner->GetInputDataTypes()[0];
                        auto data_type_size = magicmind::DataTypeSize(input_data_type);
                        auto img_data_len = img_chw*data_type_size;
                        memcpy((float*)(batch_data[0])+batch_off_set, ret.data , img_data_len);
                    }// if start >= pass_frames，开始抽帧

                }else{
                    video_end = true;
                    break;
                }// end if capture read
                start++;
            }//end while start,抽帧
            if(! video_end){
                per_vaild_batch++;
                pass_frames = clip_steps;
                clip_id++;
            }// end if video_end
        }//end for batch_size, batch data is full, start to compute

        // copy in
        cur_model_runner->H2D();
        // compute
        cur_model_runner->Run(FLAGS_batch_size);
        // copy out
        cur_model_runner->D2H();
        // get model's output addr in host
        auto host_output_ptr = cur_model_runner->GetHostOutputData();
        auto infer_res = (float *)host_output_ptr[0];
        // post process
        for(int j=0;j<per_vaild_batch;++j){
            int cur_clip_id = total_clips_id + clip_id - per_vaild_batch +j;
            std::vector<int> top5 = ArgTopK(infer_res + j*n_classes,n_classes,5);
            auto pos = test->video_paths[i].find_last_of('/');
            std::string video_name = test->video_paths[i].substr(pos+1);

            if (!FLAGS_result_label_file.empty()) {
                result_label.write("[" + std::to_string(cur_clip_id) + "]: " + std::to_string(test->name_to_id[getBaseName(test->video_paths[i])]), false);
            }
            //存放预测的top1值
            if( ! FLAGS_result_top1_file.empty()){
                result_top1_file.write("[" + std::to_string(cur_clip_id) + "]: " + std::to_string(top5[0]+1), false);
            
            }// end if top1_file
            result_file.write("top5 result in " + video_name + ":", false);
            //存放预测的top5值, key is id
            for(int k =0;k<5;++k){
                if(!FLAGS_result_top5_file.empty()){
                    result_top5_file.write("[" + std::to_string(cur_clip_id) + "]: " + std::to_string(top5[k]+1), false);
                }// end if top5

                if (!FLAGS_result_file.empty()) {
                    result_file.write(std::to_string(k) + " [" + test->id_to_name[top5[k]+1] + "]", false);
                }
            
            }// end for int k

        }//end for j
    
    }//end while video_end
    total_clips_id += clip_id;
  }//end for total_videos
  SLOG(INFO)<<"Process "<<total_videos<<" videos and "<<total_clips_id<<" clips.\n";
  // destroy resource
  cur_model_runner->Destroy();
  return 0;
}// main
