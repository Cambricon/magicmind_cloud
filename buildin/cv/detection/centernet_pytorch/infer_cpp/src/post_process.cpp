#include "../include/post_process.hpp"
#include "utils.hpp"

std::map<int, std::string> load_name(std::string name_map_file)
{
    if(!check_file_exist(name_map_file)){
        std::cout<<"coco_name file: " + name_map_file + " does not exist.\n";
        exit(0);
    }
    std::map<int, std::string> coco_name_map;
    std::ifstream in(name_map_file);
    if (!in.is_open()) { std::cout << "Open label file failed. path : " << name_map_file << std::endl;}
    if (!in){
        std::cout<<"failed to load coco_name file: " + name_map_file + ".\n";
        exit(0);
    }
    std::string line;
    int index = 0;
    while (getline(in, line))
    {
        coco_name_map[index] = line;
        index += 1;
    }
    return coco_name_map;
}

/**
 * arg top k implementation.
 * [idx_begin, idx_end): data[idx_end] will not participate in the calculation.
 * @param data[in]: input
 * @param idx_begin[in]: the index of the first data.
 * @param idx_end[in]: the index of the last data.
 * @param k[in]: output size
 * @param comp[in]: sort comparsion function
 * @return Returns the index of k data
 **/
template<typename T> static
std::vector<int64_t> ArgTopK(const T *data, int64_t idx_begin, int64_t idx_end, int64_t k,
    const std::function<bool(const T&, const T&)> &comp = std::greater<T>()) {
  if (idx_end - idx_begin < k) { std::cout << "ArgTopK: K is too large." << std::endl;}
  if (idx_end <= idx_begin) { std::cout << "ArgTopK: idx_end <= idx_begin." << std::endl;}
  std::vector<int64_t> result;
  result.reserve(k);
  int64_t i = idx_begin;
  for (; i < k + idx_begin; ++i) result.push_back(i);
  auto idx_comp = [&] (int64_t a, int64_t b) { return comp(data[a], data[b]); };
  std::make_heap(result.begin(), result.end(), idx_comp);
  for (; i < idx_end; ++i) {
    if (idx_comp(i, result[0])) {
      std::pop_heap(result.begin(), result.end(), idx_comp);
      result.back() = i;
      std::push_heap(result.begin(), result.end(), idx_comp);
    }
  }
  return result;
}

/**
 * topk implementation
 * @param data[in]: input
 * @param k[in]: output size
 * @param comp[in]: sort comparsion function
 **/
template<typename T> static
std::vector<T> TopK(const std::vector<T> &data, int64_t k,
    const std::function<bool(const T&, const T&)> &comp = std::greater<T>()) {
  if (static_cast<int64_t>(data.size()) < k) { std::cout << "TopK: K is too large." << std::endl;}
  std::vector<T> result;
  result.assign(data.begin(), data.begin() + k);
  std::make_heap(result.begin(), result.end(), comp);
  for (int64_t i = k; i < static_cast<int64_t>(data.size()); ++i) {
    if (comp(data[i], result[0])) {
      std::pop_heap(result.begin(), result.end(), comp);
      result.back() = data[i];
      std::push_heap(result.begin(), result.end(), comp);
    }
  }
  return result;
}

/**
 * @return Returns the flattened index of the k highest score in the heatmap. 
 **/
static
std::vector<int64_t> HeatMapArgTopK(const float *heatmap, const magicmind::Dims &dims, int64_t k) {
    int64_t num_cls = dims[1];
    int64_t h = dims[2], w = dims[3];
    int64_t cls_offset = h * w;
    k = std::min(k, num_cls * cls_offset);
    int64_t clsk = std::min(k, cls_offset);
  
    // topk for each class (w, h flattened)
    std::vector<int64_t> topk_idxes(num_cls * clsk);
    for (int ci = 0; ci < num_cls; ++ci) {
        auto cls_topk_idxes = ArgTopK(heatmap, ci * cls_offset, (ci + 1) * cls_offset, clsk);
        memcpy(topk_idxes.data() + ci * clsk, cls_topk_idxes.data(), cls_topk_idxes.size() * sizeof(int64_t));
    }
  
    // topk for all classes
    topk_idxes = TopK(topk_idxes, k,
        std::function<bool(const int64_t&, const int64_t&)>([&] (const int64_t &a, const int64_t &b) {
            return heatmap[a] > heatmap[b];
        }));
  
    return topk_idxes;
}

// NCHW implementation. Decode bboxes for CenterNet
// There are four outputs(heatmap_max, heatmap, wh, reg) for CenterNet
std::vector<BBox> Postprocess(const std::vector<float *> &outputs, const std::vector<magicmind::Dims> &output_dims, int max_num, float thresholds)
{
    if (output_dims.size() != 4) { std::cout << "There must be four output dimentations." << std::endl;}
    if (outputs.size() != 4) { std::cout << "There must be four outputs (heatmap_max, heatmap, wh, reg)." << std::endl;}
    float *heatmap_max = outputs[0];  // heatmap after max_pool2d
    float *heatmap = outputs[1];
    float *wh = outputs[2];
    float *reg = outputs[3];
    auto heatmap_dims = output_dims[0];
    int64_t num_cls = heatmap_dims[1];
    int64_t hm_h = heatmap_dims[2], hm_w = heatmap_dims[3];
    int64_t hm_cls_offset = hm_h * hm_w;

    // (heatmap_max == heatmap) * heatmap
    cv::Mat hm_max_mat(heatmap_dims.GetElementCount() / heatmap_dims[0], 1, CV_32FC1, heatmap_max);
    cv::Mat hm_mat(heatmap_dims.GetElementCount() / heatmap_dims[0], 1, CV_32FC1, heatmap);
    cv::Mat keep;
    cv::compare(hm_max_mat, hm_mat, keep, cv::CMP_EQ);  // hm = (hm_max == hm) ? 255 : 0
    keep.convertTo(keep, CV_32F, 1.0 / 255);
    hm_mat = hm_mat.mul(keep);

    std::vector<int64_t> topk_idxes = HeatMapArgTopK(heatmap, heatmap_dims, max_num);
    std::vector<BBox> bboxes;
    for (auto flattened_idx : topk_idxes) {
        BBox bbox;
        bbox.label = flattened_idx / (hm_h * hm_w);
        bbox.score = heatmap[flattened_idx];
        if (bbox.score < thresholds) continue;
        int64_t hm_xy_flattened_idx = flattened_idx % (hm_cls_offset);
        int64_t hm_y = hm_xy_flattened_idx / hm_w;
        int64_t hm_x = hm_xy_flattened_idx % hm_w;
        int64_t w_offset = hm_y * hm_w + hm_x;
        int64_t h_offset = hm_y * hm_w + hm_x + hm_cls_offset;
        int w = static_cast<int>(wh[w_offset]);
        int h = static_cast<int>(wh[h_offset]);
        bbox.left = static_cast<int>(hm_x + reg[w_offset] - std::round(w / 2.0));
        bbox.top = static_cast<int>(hm_y + reg[h_offset] - std::round(h / 2.0));
        bbox.right = bbox.left + w;
        bbox.bottom = bbox.top + h;
        bboxes.push_back(bbox);
    }
    return bboxes;
}

/**
 * Rescale bounding-boxes to origin image.
 * @param img[in]: rescale to specified image.
 * @param heatmap_dims[in]: the heatmap dimentations.
 * @param bboxes[in and out]: gets from CenterNetPostprocess.
 */
void RescaleBBox(
    cv::Mat img, const magicmind::Dims &heatmap_dims,
    std::vector<BBox> &bboxes, std::map<int, std::string> imagenet_name_map, const std::string name, const std::string output_dir) {
    std::string filename = output_dir + "/" + name + ".txt";
    std::ofstream file_map(filename);
    float h_scale = 1.0f * img.rows / heatmap_dims[2];
    float w_scale = 1.0f * img.cols / heatmap_dims[3];
    for (auto iter = bboxes.begin(); iter != bboxes.end();) {
        auto &bbox = *iter;
        bbox.left = std::floor(bbox.left * w_scale);
        bbox.top = std::floor(bbox.top * h_scale);
        bbox.right = std::floor(bbox.right * w_scale);
        bbox.bottom = std::floor(bbox.bottom * h_scale);
        bbox.left = std::max(bbox.left, 0);
        bbox.top = std::max(bbox.top, 0);
        bbox.right = std::min(bbox.right, img.cols);
        bbox.bottom = std::min(bbox.bottom, img.rows);
        if (bbox.right <= bbox.left || bbox.bottom <= bbox.top) {
            iter = bboxes.erase(iter);
        } else {
            iter++;
            file_map << imagenet_name_map[bbox.label] << "," 
                 << bbox.score << ","
                 << bbox.left << ","
                 << bbox.top << ","
                 << bbox.right << ","
                 << bbox.bottom << "\n"; 
        }
    }
    file_map.close();
}

void Draw(cv::Mat img, const std::vector<BBox> &bboxes, std::map<int, std::string> imagenet_name_map) {
  for (const auto &bbox : bboxes) {
    cv::rectangle(img, cv::Point(bbox.left, bbox.top),
        cv::Point(bbox.right, bbox.bottom), cv::Scalar(0, 255, 0), 2);
    std::string text = imagenet_name_map[bbox.label] + ": " + std::to_string(bbox.score);
    cv::putText(img, text, cv::Point(bbox.left, bbox.top + 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 0);
  }
}

