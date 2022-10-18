#include "../include/utils.hpp"
#include "../include/pre_process.hpp"
/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
result * LoadImages(const std::string image_dir, int batch_size, int image_num, const std::string file_list) {
    result *test=new result;
    char abs_path[PATH_MAX];
    if (realpath(image_dir.c_str(), abs_path) == NULL) {
        std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
        exit(1);
    }
    std::string glob_path = std::string(abs_path);
    std::ifstream in(file_list);
    std::string line;
    std::string image_name;
    int count = 0;
    std::string image_path;
    int label;
    while(getline(in, line)) {
        int found = line.find(" ");
        image_name = line.substr(0, found);
        label = std::stoi(line.substr(found+1));
        image_path = glob_path + "/" +image_name;
        test->image_paths.push_back(image_path);
        test->labels.push_back(label);
        count += 1;
        if (count >= image_num) {break;}
    }
    // pad to multiple of batch_size.
    // The program will stuck when the number of input images is not an integer multiple of the batch size
    size_t pad_num = batch_size - test->image_paths.size() % batch_size;
    if (pad_num != batch_size) {
        std::cout << "There are " << test->image_paths.size() << " images in total, add " << pad_num
            << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
        while (pad_num--)
        test->image_paths.emplace_back(*test->image_paths.rbegin());
    }
    return test;
}

DEFINE_int32(new_size, 256, "The resized image size.");
DEFINE_bool(input_rgb, true, "Input convert to rgb or not.");

// resize to new_size + center crop to input_dim
cv::Mat Preprocess(cv::Mat img, const magicmind::Dims &input_dim) {
    // NHWC order implementation. Make sure your model's input is in NHWC order.
    /*
        (x - mean) / std : This calculation process is performed at the first layer of the model,
        See parameter named [insert_bn_before_firstnode] in magicmind::IBuildConfig.
    */
    int h = input_dim[1];
    int w = input_dim[2];
    if (h > FLAGS_new_size || w > FLAGS_new_size) {
        std::cout << "new_size[" << FLAGS_new_size << "] less than input_dim[" << input_dim << "]." << std::endl;
    }
    // resize
    float scale = 1.0f * FLAGS_new_size / std::min(img.cols, img.rows);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(std::round(scale * img.cols), std::round(scale * img.rows)));
    // center crop
    auto roi = resized(cv::Rect((resized.cols - w) / 2, (resized.rows - h) / 2, w, h));
    cv::Mat dst_img(h, w, CV_8UC3, cv::Scalar(0, 0, 255));
    if (FLAGS_input_rgb) {
        cv::cvtColor(roi, dst_img, cv::COLOR_BGR2RGB);
    } else {
        roi.copyTo(dst_img);
    }
    return dst_img;
}

