#include "../include/pre_process.hpp"
#include "../include/utils.hpp"

/**
 * @brief load all images(jpg) from image directory(args.image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir,int batch_size) {
    char abs_path[PATH_MAX];
    if (realpath(image_dir.c_str(), abs_path) == NULL) {
            std::cout << "Get real image path in " << image_dir.c_str() << " failed...";
            exit(1);
    }
    std::string glob_path = std::string(abs_path);
    std::ifstream in(image_dir+"/VOC2007/ImageSets/Main/test.txt");
    std::string image_name;
    std::string image_path;
    std::vector<std::string> image_paths;
    while(getline(in, image_name)) {
        image_path = glob_path + "/VOC2007/JPEGImages/" + image_name + ".jpg";
        image_paths.push_back(image_path);
    }
    // pad to multiple of batch_size.
    // The program will stuck when the number of input images is not an integer multiple of the batch size
    size_t pad_num = batch_size - image_paths.size() % batch_size;
    if (pad_num != batch_size) {
        std::cout << "There are " << image_paths.size() << " images in total, add " << pad_num
            << " more images to make the number of images is an integral multiple of batchsize[" << batch_size << "].";
        while (pad_num--){
            image_paths.emplace_back(*image_paths.rbegin());
        }
    }
    return image_paths;
}

cv::Mat process_img(cv::Mat src_img, bool transpose, bool normlize, bool swapBR, int depth){
    int src_h = src_img.rows;
    int src_w = src_img.cols;
    int dst_h = 300;
    int dst_w = 300;

    cv::resize(src_img, src_img, cv::Size(dst_w, dst_h), cv::INTER_LINEAR);
    
    if (normlize)
    {
        src_img.convertTo(src_img, CV_32F);
        cv::Scalar mean(127.5, 127.5, 127.5);
        cv::Scalar std(0.007843, 0.007843, 0.007843);
        cv::subtract(src_img, mean, src_img);
        cv::multiply(src_img, std, src_img);
    }
    if (swapBR)
    {
        cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    }

    cv::Mat blob;
    if (transpose)
    {
        int c = src_img.channels();
        int h = src_img.rows;
        int w = src_img.cols;
        int sz[] = {1, c, h, w};
        blob.create(4, sz, depth);
        cv::Mat ch[3];
        for (int j = 0; j < c; j++)
        {
            ch[j] = cv::Mat(src_img.rows, src_img.cols, depth, blob.ptr(0, j));
        }
        cv::split(src_img, ch);
    }
    else
    {
        blob = src_img;
    }

    return blob;
}

