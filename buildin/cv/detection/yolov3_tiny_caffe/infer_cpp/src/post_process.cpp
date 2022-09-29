#include "post_process.hpp"
#include "utils.hpp"

std::vector<BBox> Yolov3GetBBox(cv::Mat img, float scaling_factors,
                                const int input_h, const int input_w,
                                float *bbox_output, int bbox_num,
                                float confidence_thresholds) {
  static constexpr int bbox_size = 7;  // every 7 values form a bounding box.
  //[ label score x1 y1 x2 y2 ]
  std::vector<BBox> bboxes;
  float pad_top = (input_h - scaling_factors * img.rows) / 2;
  float pad_left = (input_w - scaling_factors * img.cols) / 2;
  for (int i = 0; i < bbox_num; ++i) {
    float *bbox_data = bbox_output + i * bbox_size;
    if (confidence_thresholds > bbox_data[2]) continue;
    BBox bbox = {
        static_cast<int>(bbox_data[1]),  // category id
        bbox_data[2],                    // score
        static_cast<int>(std::floor((bbox_data[3] * input_w - pad_left) /
                                    scaling_factors)),  // left
        static_cast<int>(std::floor((bbox_data[4] * input_h - pad_top) /
                                    scaling_factors)),  // top
        static_cast<int>(std::floor((bbox_data[5] * input_w - pad_left) /
                                    scaling_factors)),  // right
        static_cast<int>(std::floor((bbox_data[6] * input_h - pad_top) /
                                    scaling_factors))  // bottom
    };
    if (bbox.left >= bbox.right || bbox.top >= bbox.bottom) continue;
    bbox.left = bbox.left > 0 ? bbox.left : 0;
    bbox.right = bbox.right < img.cols ? bbox.right : img.cols;
    bbox.top = bbox.top > 0 ? bbox.top : 0;
    bbox.bottom = bbox.bottom < img.rows ? bbox.bottom : img.rows;
    bboxes.emplace_back(bbox);
  }
  return bboxes;
}

// draw bboxes on image
void Draw(cv::Mat img, const std::vector<BBox> &bboxes,
          const std::vector<std::string> &labels) {
  for (const auto &bbox : bboxes) {
    cv::rectangle(img, cv::Point(bbox.left, bbox.top),
                  cv::Point(bbox.right, bbox.bottom), cv::Scalar(0, 255, 0), 2);
    std::string text =
        (bbox.label < labels.size() ? labels[bbox.label] : "Unknow label") +
        ":" + std::to_string(std::round(bbox.score * 1000) / 1000.0f);
    cv::putText(img, text, cv::Point(bbox.left, bbox.top + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 0);
  }
}

static int LabelNameToCategoryId(const std::string &label_name) {
  auto iter = kCOCOPaperCategoryIdMap.find(label_name);
  return kCOCOPaperCategoryIdMap.end() == iter ? -1 : iter->second;
}

void saveImgAndPreds(bool save_img_, bool save_pred_,
                     vector<cv::Mat> &src_imgs_, vector<string> &image_name_,
                     const string &save_img_dir_, const string &save_pred_dir_,
                     const std::vector<std::string> &labels_, const int num_,
                     const vector<BBox> &bboxes_) {
  if (save_img_) {
    Draw(src_imgs_[num_], bboxes_, labels_);
    std::string save_path =
        save_img_dir_ + "/" + GetFileName(image_name_[num_]) + "_drawed.jpg";
    cv::imwrite(save_path, src_imgs_[num_]);
  }
  if (save_pred_) {
    string base_file_name = GetFileName(image_name_[num_]);
    string save_pred_filename = save_pred_dir_ + '/' + base_file_name + ".txt";
    std::ofstream file_map(save_pred_filename);
    int image_id = stoi(base_file_name);
    for (int b_i = 0; b_i < bboxes_.size(); b_i++) {
      BBox cur_box = bboxes_[b_i];
      std::string label_name = labels_[cur_box.label];
      //根据compute_cooo_mAP.py要求所得
      file_map << label_name << "," << cur_box.score << ","
               << static_cast<int>(cur_box.left) << ","
               << static_cast<int>(cur_box.top) << ","
               << static_cast<int>(cur_box.right) << ","
               << static_cast<int>(cur_box.bottom) << "\n";
    }
    file_map.close();
  }
}
