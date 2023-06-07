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
#include <float.h>

#include "pre_process.hpp"
#include "post_process.hpp"
#include "utils.hpp"
#include "model_runner.h"

using namespace magicmind;
using namespace std;
using namespace cv;

DEFINE_int32(device_id, 0, "The device index of mlu");
DEFINE_string(magicmind_model, "", "The magicmind model path");
DEFINE_string(image_dir, "", "The image directory");
DEFINE_int32(image_num, 1, "image number");
DEFINE_string(file_list, "coco_file_list_5000.txt", "file_list");
DEFINE_string(label_path, "coco.names", "The label path");
DEFINE_int32(max_bbox_num, 100, "Max number of bounding-boxes per image");
DEFINE_double(confidence_thresholds, 0.001, "");
DEFINE_string(output_dir, "", "../data/images");
DEFINE_bool(save_img, false, "whether saving the image or not");
DEFINE_int32(batch_size, 1, "The batch size");

int main(int argc, char **argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    TimeCollapse time_centernet("infer centernet"); 

    auto centernet_runner = new ModelRunner(FLAGS_device_id, FLAGS_magicmind_model);
    if (!centernet_runner->Init(FLAGS_batch_size)) {
    	SLOG(ERROR) << "Init yolov5 runnner failed.";
    	return false;
    }

 
    //  load image
    std::cout << "================== Load Images ====================" << std::endl;
    std::vector<std::string> image_paths = LoadImages(FLAGS_image_dir, FLAGS_batch_size, FLAGS_image_num, FLAGS_file_list);
    if (image_paths.size() == 0) {
       std::cout << "No images found in dir [" << FLAGS_image_dir << "]. Support jpg.";
       return 0;
    }
    size_t image_num = image_paths.size();
    size_t rem_image_num = image_num % FLAGS_batch_size;
    SLOG(INFO) << "Total images : " << image_num << std::endl;
    std::map<int, std::string> name_map = load_name(FLAGS_label_path);

    // batch information
    int batch_counter = 0;
    std::vector<std::string> batch_image_name;
    std::vector<cv::Mat> batch_image;
    
    // allocate host memory for batch preprpcessed data
    auto batch_data = centernet_runner->GetHostInputData();

    // one batch input data addr offset
    int batch_image_offset = centernet_runner->GetInputSizes()[0] / FLAGS_batch_size;

    auto input_dim_vec = centernet_runner->GetInputDims()[0];
    magicmind::Dims input_dim = magicmind::Dims(input_dim_vec);
    //int h = input_dim[1];
    //int w = input_dim[2];
    
    SLOG(INFO) << "Start run..." << std::endl;  
    for (int i = 0 ; i < image_num ; i ++) {
      string image_name = image_paths[i].substr(image_paths[i].find_last_of('/') + 1, 12);
      std::cout << "Inference img : " << image_name << "\t\t\t" << i+1 << "/" << image_num << std::endl;
      cv::Mat img = cv::imread(image_paths[i]);
      cv::Mat img_pro = Preprocess(img, input_dim);
      batch_image_name.push_back(image_name);
      batch_image.push_back(img);

      memcpy((u_char *)(batch_data[0]) + batch_counter * batch_image_offset, img_pro.data,batch_image_offset);
      batch_counter += 1;

      // image_num may not be divisible by FLAGS_batch.
      // real_batch_size records number of images in every loop, real_batch_size may change in the
      // last loop.
      size_t real_batch_size = (i < image_num - rem_image_num) ? FLAGS_batch_size : rem_image_num;
      if (batch_counter % real_batch_size == 0) {
      	// copy in
      	centernet_runner->H2D();
      	// compute
      	centernet_runner->Run(FLAGS_batch_size);
      	// copy out
      	centernet_runner->D2H();
      // get model's output addr in host
      	auto host_output_ptr = centernet_runner->GetHostOutputData();

        vector<magicmind::DataType> vec_box_dtype;
        for(int tt = 0; tt < centernet_runner->GetOutputDataTypes().size(); tt++) {
            magicmind::DataType box_dtype = centernet_runner->GetOutputDataTypes()[tt];
            vec_box_dtype.push_back(box_dtype);
        }
        vector<int> vec_batch_box_offset;
        for(int tt = 0; tt < centernet_runner->GetOutputSizes().size(); tt++) {
            int batch_box_offset = centernet_runner->GetOutputSizes()[tt] / FLAGS_batch_size / sizeof(vec_box_dtype[tt]);
            vec_batch_box_offset.push_back(batch_box_offset);
        }
        auto output_dims_vec = centernet_runner->GetOutputDims();
        vector<magicmind::Dims> mm_output_dims_vec;
        for(int tt = 0;tt <output_dims_vec.size();tt++) {
          output_dims_vec[tt][0] = output_dims_vec[tt][0]/real_batch_size;
          magicmind::Dims output_dims = magicmind::Dims(output_dims_vec[tt]);
          mm_output_dims_vec.push_back(output_dims);
        }
        std::vector<float*> net_outputs(host_output_ptr.size());
        std::vector<float*> temp(host_output_ptr.size());
        
        for(int j=0;j<real_batch_size;j++) {
          for (size_t tt = 0; tt < host_output_ptr.size(); ++tt) {
            temp[tt] = static_cast<float*>(host_output_ptr[tt]);
            net_outputs[tt] = temp[tt] + j*vec_batch_box_offset[tt];
          }
	        auto bboxes = Postprocess(net_outputs, mm_output_dims_vec, FLAGS_max_bbox_num, FLAGS_confidence_thresholds);
          RescaleBBox(batch_image[j], mm_output_dims_vec[0], bboxes, name_map, batch_image_name[j], FLAGS_output_dir);
          if (FLAGS_save_img) {
            // draw bboxes on original image and save it to disk.
            cv::Mat origin_img = batch_image[j].clone();
            Draw(batch_image[j], bboxes, name_map);
            cv::imwrite(FLAGS_output_dir + "/" + batch_image_name[j] + ".jpg", batch_image[j]);
          }
        }
	      
        batch_counter = 0;
        batch_image.clear();
        batch_image_name.clear();
      }
    } 
    centernet_runner->Destroy();
    return 0;
}
