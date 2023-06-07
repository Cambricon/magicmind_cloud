#ifndef PRE_PROCESS_HPP
#define PRE_PROCESS_HPP

#include <gflags/gflags.h>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include <deque>
using namespace std;

struct result{
    std::vector<std::string> video_paths;
    std::vector<int> labels;
    std::map<int,std::string> id_to_name;
    std::map<std::string,int> name_to_id;
};
cv::Mat PreprocessImage(cv::Mat img);
result* loadVideosAndLabels(const std::string &path,const std::string &label_file);
std::string getBaseName(const string& fullname);
#endif //PRE_PROCESS_HPP
