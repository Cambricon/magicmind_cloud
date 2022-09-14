#include "../include/pre_process.hpp"
#include "../include/utils.hpp"

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir, int batch_size, int image_num, const std::string file_list) {
    char abs_path[PATH_MAX];
    if (realpath(image_dir.c_str(), abs_path) == NULL) {
        std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
        exit(1);
    }
    std::string glob_path = std::string(abs_path);
    std::ifstream in(file_list);
    std::string image_name;
    std::vector<std::string> image_paths;
    int count = 0;
    std::string image_path;
    while(getline(in, image_name)) {
        image_path = glob_path + "/" +image_name;
        image_paths.push_back(image_path);
        count += 1;
        if (count >= image_num) {break;}
    }
    // pad to multiple of batch_size.
    // The program will stuck when the number of input images is not an integer multiple of the batch size
    size_t pad_num = batch_size - image_paths.size() % batch_size;
    if (pad_num != batch_size) {
        std::cout << "There are " << image_paths.size() << " images in total, add " << pad_num
            << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
        while (pad_num--)
            image_paths.emplace_back(*image_paths.rbegin());
    }
    return image_paths;
}

cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim){
    // NHWC order implementation. Make sure your model's input is in NHWC order.
    /*
       (x - mean) / std : This calculation process is performed at the first layer of the model,
       See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
    */
    //resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_dim[2], input_dim[1]));
    //bgr to rgb
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    return resized;
}

