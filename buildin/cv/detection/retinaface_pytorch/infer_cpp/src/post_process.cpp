#include <mm_runtime.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <utility>
#include <vector>
#include <gflags/gflags.h>

#include "../include/post_process.hpp"
#include "utils.hpp"

#define CHECK_EQ(...)

// network config
static std::vector<std::pair<int, int>> gkMinSizes = {{16, 32}, {64, 128}, {256, 512}};
static std::vector<int> gkSteps = {8, 16, 32};
static std::array<float, 2> gkVariances = {0.1f, 0.2f};

struct RawBBox {
  float cx, cy, w, h;
  float score;
  std::vector<cv::Point2f> landms;
};  // struct RawBBox

static std::vector<RawBBox> PriorBox(int height, int width) {
  std::vector<std::pair<float, float>> feature_maps;
  for (auto step : gkSteps) {
    feature_maps.emplace_back(
        std::make_pair(std::ceil(1.0 * height / step), std::ceil(1.0 * width / step)));
  }
  std::vector<RawBBox> anchors;
  for (size_t fm_id = 0; fm_id < feature_maps.size(); ++fm_id) {
    std::vector<int> min_size = {gkMinSizes[fm_id].first, gkMinSizes[fm_id].second};
    const auto &fm = feature_maps[fm_id];
    for (int i = 0; i < fm.first; ++i) {
      for (int j = 0; j < fm.second; ++j) {
        for (auto size : min_size) {
          float s_kx = 1.0f * size / width;
          float s_ky = 1.0f * size / height;
          float cx = (j + 0.5) * gkSteps[fm_id] / width;
          float cy = (i + 0.5) * gkSteps[fm_id] / height;
          anchors.emplace_back(RawBBox({cx, cy, s_kx, s_ky, 0.f, {}}));
        }  // for min_size
      }    // for fm.second
    }      // for fm.first
  }        // for feature_maps
  return anchors;
}

static std::vector<RawBBox> gkAnchors = {};

// decode bboxes(center xywh, score, landms)
static std::vector<RawBBox> Decode(const std::vector<const float *> &preds,
                                   const std::vector<std::vector<int64_t>> &dims,
                                   float thresh) {
  CHECK_EQ(preds.size(), 3);
  CHECK_EQ(dims.size(), 3);
  auto loc = preds[0];
  auto conf = preds[1];
  auto landms = preds[2];
  auto loc_dim = dims[0];
  auto conf_dim = dims[1];
  auto landms_dim = dims[2];
  const int num_landms = landms_dim[2] / 2;
  // output dims: batchsize num_bboxes 4
  CHECK_EQ(loc_dim[1], gkAnchors.size());  // shape immutable case, number of anchors never changed.
  CHECK_EQ(conf_dim[1], gkAnchors.size());
  CHECK_EQ(landms_dim[1], gkAnchors.size());
  CHECK_EQ(loc_dim[2], 4);
  CHECK_EQ(conf_dim[2], 2);
  CHECK_EQ(landms_dim[2] % 2, 0);
  std::vector<RawBBox> bboxes;
  bboxes.reserve(gkAnchors.size());
  for (size_t i = 0; i < gkAnchors.size(); ++i) {
    const auto &anchor = gkAnchors[i];
    const float *loct = loc + i * 4;
    const float *conft = conf + i * 2;
    const float *landmst = landms + i * 2 * num_landms;
    if (conft[1] < thresh)
      continue;
    RawBBox bbox;
    // center xywh
    bbox.cx = anchor.cx + loct[0] * gkVariances[0] * anchor.w;
    bbox.cy = anchor.cy + loct[1] * gkVariances[0] * anchor.h;
    bbox.w = anchor.w * std::exp(loct[2] * gkVariances[1]);
    bbox.h = anchor.h * std::exp(loct[3] * gkVariances[1]);
    // landms
    bbox.landms.resize(landms_dim[2] / 2);
    for (int landm_id = 0; landm_id < num_landms; ++landm_id) {
      bbox.landms[landm_id].x = anchor.cx + landmst[landm_id * 2] * gkVariances[0] * anchor.w;
      bbox.landms[landm_id].y = anchor.cy + landmst[landm_id * 2 + 1] * gkVariances[0] * anchor.h;
    }
    bbox.score = conft[1];
    bboxes.emplace_back(bbox);
  }
  return bboxes;
}

// rescale bboxes to origin image
static std::vector<BBox> RescaleBBox(const std::vector<RawBBox> &raw_bboxes,
                                     std::vector<int64_t> &input_dim,
                                     cv::Mat img,
                                     float scaling_factor) {
  int input_h = input_dim[1];
  int input_w = input_dim[2];
  int img_h = img.rows;
  int img_w = img.cols;
  int pad_left = std::floor((input_w - std::floor(img_w * scaling_factor)) / 2);
  int pad_top = std::floor((input_h - std::floor(img_h * scaling_factor)) / 2);
  std::vector<BBox> bboxes;
  bboxes.reserve(raw_bboxes.size());
  for (const auto &raw_bbox : raw_bboxes) {
    BBox bbox;
    bbox.cx = std::floor((raw_bbox.cx * input_w - pad_left) / scaling_factor);
    bbox.cy = std::floor((raw_bbox.cy * input_h - pad_top) / scaling_factor);
    bbox.w = std::round(raw_bbox.w * input_w / scaling_factor);
    bbox.h = std::round(raw_bbox.h * input_h / scaling_factor);
    bbox.landms.reserve(raw_bbox.landms.size());
    if (bbox.cx < 0 || bbox.cy < 0 || bbox.w < 0 || bbox.h < 0)
      continue;  // invalid bbox
    for (const auto &raw_landm : raw_bbox.landms) {
      auto landm = cv::Point2f(std::floor((raw_landm.x * input_w - pad_left) / scaling_factor),
                               std::floor((raw_landm.y * input_h - pad_top) / scaling_factor));
      if (landm.x < 0.f || landm.y < 0.f)
        continue;
      bbox.landms.emplace_back(landm);
    }
    bbox.score = raw_bbox.score;
    bboxes.emplace_back(bbox);
  }
  return bboxes;
}

static float IOU(const BBox &a, const BBox &b) {
  float area_a = (a.w + 1) * (a.h + 1);
  float area_b = (b.w + 1) * (b.h + 1);
  float left = std::max(a.cx - a.w / 2.0f, b.cx - b.w / 2.0f);
  float right = std::min(a.cx + a.w / 2.0f, b.cx + b.w / 2.0f);
  float top = std::max(a.cy - a.h / 2.0f, b.cy - b.h / 2.0f);
  float bottom = std::min(a.cy + a.h / 2.0f, b.cy + b.h / 2.0f);
  float overlap_area = std::max(0.f, right - left + 1) * std::max(0.f, bottom - top + 1);
  return overlap_area / (area_a + area_b - overlap_area);
}

static std::vector<BBox> NMS(std::vector<BBox> &&preds, float nms_thresh) {
  std::sort(preds.begin(), preds.end(),
            [](const BBox &a, const BBox &b) { return a.score < b.score; });
  std::vector<BBox> results;
  while (!preds.empty()) {
    results.emplace_back(preds.back());
    preds.pop_back();
    auto iter = preds.begin();
    while (preds.end() != iter) {
      float iou = IOU(results.back(), *iter);
      if (iou > nms_thresh)
        iter = preds.erase(iter);
      else
        iter++;
    }
  }
  return results;
}

void InitAnchors(int height, int width) {
  gkAnchors = PriorBox(height, width);
}

DEFINE_double(confidence_thresholds, 0.02, "Confidence thresholds");
DEFINE_double(nms_thresholds, 0.4, "NMS thresholds");

std::vector<BBox> Postprocess(cv::Mat img,
                              std::vector<int64_t> &input_dim,
                              std::vector<const float *> outputs,
                              std::vector<std::vector<int64_t>> output_dims) {
  float scaling_factors = std::min(1.0f * input_dim[1] / img.rows, 1.0f * input_dim[2] / img.cols);
  InitAnchors(input_dim[1], input_dim[2]);
  auto preds = Decode(outputs, output_dims, FLAGS_confidence_thresholds);
  auto bboxes_before_nms = RescaleBBox(preds, input_dim, img, scaling_factors);
  auto bboxes = NMS(std::move(bboxes_before_nms), FLAGS_nms_thresholds);

  return bboxes;
}

std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  if (std::string::npos == slash_pos) {
    std::cout << "[" << abs_path << "] is not an absolute path." << std::endl;
    return "";
  }
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  if (point_pos == slash_pos + 1) {
    std::cout << "[" << abs_path << "] is not a file path." << std::endl;
    return "";
  }
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}