#include "../include/post_process.hpp"
#include "../include/utils.hpp"

bool post_process(cv::Mat &img, std::vector<std::vector<float>> results, std::vector<std::string> imagenet_name_map, std::vector<std::string> voc_preds_files, const std::string name, const std::string output_dir, bool save_img)
{
    int src_h = img.rows;
    int src_w = img.cols;
    int dst_h = 300;
    int dst_w = 300;
    float ratio_h = float(src_h)/float(dst_h);
    float ratio_w = float(src_w)/float(dst_w);

    int detect_num = results.size();
    for (int i = 0; i < detect_num; ++i){
        int detect_class = results[i][0];
        float score = results[i][1];
        float xmin = results[i][2];
        float ymin = results[i][3];
        float xmax = results[i][4];
        float ymax = results[i][5];
        if (0 == detect_class) continue; //background
        xmin = xmin * ratio_w * dst_w;
        ymin = ymin * ratio_h * dst_h;
        xmax = xmax * ratio_w * dst_w;
        ymax = ymax * ratio_h * dst_h;
        if (score < 0.01) continue;
        if (xmin >= xmax || ymin >= ymax) continue;
        xmin = std::max(float(0), xmin);
        xmax = std::min(xmax, float(src_w));
        ymin = std::max(float(0), ymin);
        ymax = std::min(ymax, float(src_h));
        std::string filename = voc_preds_files[detect_class];
        std::ofstream file_map(filename, std::ios::app);
        file_map << name << " " 
                 << score << " "
                 << xmin << " "
                 << ymin << " "
                 << xmax << " "
                 << ymax << "\n"; 
        file_map.close();
        if (save_img) {
            cv::rectangle(img, cv::Rect(cv::Point(int(xmin), int(ymin)), cv::Point(int(xmax), int(ymax))), cv::Scalar(0, 255, 0));
            auto fontface = cv::FONT_HERSHEY_TRIPLEX;
            double fontscale = 0.5;
            int thickness = 1;
            int baseline = 0;
            std::string text = imagenet_name_map[detect_class] + ": " + std::to_string(score);
            cv::Size text_size = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
            cv::putText(img, text, cv::Point(int(xmin), int(ymin) + text_size.height), fontface, fontscale, cv::Scalar(255, 255, 255), thickness);
        }
    }
    if (save_img) {
      imwrite(output_dir + "/" + name + ".jpg", img);
    }
    return true;
}