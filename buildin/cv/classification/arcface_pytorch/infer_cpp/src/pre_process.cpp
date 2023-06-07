#include "pre_process.h"
#include "face_align.h"

/**
 * @brief load all images(jpg) from image directory(FLAGS_image_dir)
 * @return Returns image paths
 */
std::vector<std::string> LoadImages(const std::string image_dir,
                                    const std::string image_list,
                                    int image_num,
                                    const int batch_size) {
  std::vector<std::string> image_paths = LoadFileList(image_list, image_num);
  // pad to multiple of batch_size.
  // The program will stuck when the number of input images is not an integer multiple of the batch
  // size size_t pad_num = batch_size - image_paths.size() % batch_size; if (pad_num != batch_size)
  // {
  //   LOG(INFO) << "There are " << image_paths.size() << " images in total, add " << pad_num
  //       << " more images to make the number of images is an integral multiple of batchsize[" <<
  //       batch_size << "].";
  //   while (pad_num--)
  //     image_paths.emplace_back(*image_paths.rbegin());
  // }
  return image_paths;
}

cv::Mat Preprocess(cv::Mat img,
                   std::vector<int64_t> &input_dim,
                   const std::vector<std::string> landmarks) {
  int h = input_dim[1];
  int w = input_dim[2];
  static constexpr float src_data[10] = {30.2946f + 8.0f, 51.6963f, 65.5318f + 8.0f, 51.5014f,
                                         48.0252f + 8.0f, 71.7366f, 33.5493f + 8.0f, 92.3655f,
                                         62.7299f + 8.0f, 92.2041f};  // +8.0f for 112*112
  cv::Mat src(5, 2, CV_32FC1, const_cast<float *>(src_data));
  float landmarks_data[10] = {0};
  for (int index = 0; index < 10; index++) {
    landmarks_data[index] = (float)std::atof(landmarks[index].c_str());
  }
  cv::Mat landmark_mat(5, 2, CV_32FC1, reinterpret_cast<float *>(landmarks_data));

  cv::Mat matix = similarTransform(landmark_mat,
                                   src);  // skimage.transform.SimilarityTransform
  cv::Mat M(matix, cv::Rect(0, 0, 3, 2));
  cv::Mat bgr;
  cv::warpAffine(img, bgr, M, cv::Size(w, h));
  cv::Mat rgb(h, w, CV_8UC3);
  cv::cvtColor(bgr, rgb, CV_BGR2RGB);
  return rgb;
}
