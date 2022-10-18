#include "post_process.h"
#include <glog/logging.h>
#include <mm_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "net_params.h"
#include "utils.h"

// maximumpositivion: nms 0.02, paf 0.01 sample over 0.75 subset score 0.02, subset cnt 2
// thresholds
static constexpr float kNMSThreshold = 0.05f;
static constexpr float kPAFThreshold = 0.05f;
static constexpr float kMinAboveThreshold = 0.95f;
static constexpr float kSubsetThreshold = 0.4f;
static constexpr int   kMinSubsetCount = 3;
static constexpr int   kMaxPeaks = 128;
static constexpr float kNMSOffset = 0.5;

// rescale keypoints to origin image
static
void RescaleKeypoints(Keypoints *keypoints, const float scaling_factor) {
  for (auto &points : *keypoints) {
    for (auto &point : points) {
      auto &pos = point.pos;
      pos.x /= scaling_factor;
      pos.y /= scaling_factor;
    }  // for points
  }  // for body parts
}

/**
 * @brief Post process for body pose
 */
class PostprocPose {
 public:
  PostprocPose() : net_params_(GetNetParams()) {}
  std::pair<Keypoints, PersonInfos> Execute(
      cv::Mat img, float *preds, const float scaling_factor,
      const magicmind::Dims &input_dim, const magicmind::Dims &output_dim);

 private:
  struct BodyPairInfo {
    cv::Point idxa, idxb;  // index in keypoints vector
    int body_part_pair_idx;  // index in NetParams::body_part_pairs
    float score;  // paf score
    float total_score;  // only used for sort body pairs
  };  // struct BodyPairInfo
  using BodyPairs = std::vector<BodyPairInfo>;
  using Heatmaps = std::vector<cv::Mat>;

 private:
  Heatmaps GetHeatmaps(
      float *preds, const magicmind::Dims &input_dim, const magicmind::Dims &output_dim);
  Keypoints GetKeypoints(
      const Heatmaps &heatmaps, const float scaling_factor);
  BodyPairs GetBodyPairs(
      const Heatmaps &heatmaps, const Keypoints &keypoints);
  PersonInfos GetPersonInfos(
      const Keypoints &keypoints, const BodyPairs &body_pairs);

 private:
  cv::Mat NMSRegisterKernel(const cv::Mat &confidence_map);
  std::vector<KeypointInfo> AccuratePeakPositions(
      const cv::Mat &confidence_map, const cv::Mat &nms_kernel, const float scaling_factor);
  float GetPAFScore(const KeypointInfo &a, const cv::Mat &pafa, const KeypointInfo &b, const cv::Mat &pafb);
  int GetNumberBodyPairs(const PersonInfo &person);

 private:
  const NetParams &net_params_;
};  // class PostprocPose

std::pair<Keypoints, PersonInfos>
PostprocPose::Execute(cv::Mat img, float *preds, const float scaling_factor,
    const magicmind::Dims &input_dim, const magicmind::Dims &output_dim) {
  // check model output size
  LOG_IF(FATAL, output_dim[1] != net_params_.nheatmaps)
    << "The number of heatmaps in model mismatched.";
  auto heatmaps = GetHeatmaps(preds, input_dim, output_dim);
  auto keypoints = GetKeypoints(heatmaps, scaling_factor);
  auto body_pairs = GetBodyPairs(heatmaps, keypoints);
  auto persons = GetPersonInfos(keypoints, body_pairs);
  RescaleKeypoints(&keypoints, scaling_factor);
  return std::make_pair(std::move(keypoints), std::move(persons));
}

PostprocPose::Heatmaps
PostprocPose::GetHeatmaps(
    float *preds, const magicmind::Dims &input_dim, const magicmind::Dims &output_dim) {
  Heatmaps heatmaps(net_params_.nheatmaps);
  const int src_w = output_dim[3];
  const int src_h = output_dim[2];
  const int src_heatmap_len = src_w * src_h;
  const cv::Size dst_size(input_dim[2], input_dim[1]);
  for (int i = 0; i < net_params_.nheatmaps; ++i) {
    // resize to model input size
    cv::Mat src(src_h, src_w, CV_32FC1, preds + i * src_heatmap_len);
    cv::resize(src, heatmaps[i], dst_size, 0, 0, cv::INTER_CUBIC);
  }
  return heatmaps;
}

Keypoints
PostprocPose::GetKeypoints(const Heatmaps &heatmaps, const float scaling_factor) {
  // NMS
  Keypoints keypoints;
  keypoints.reserve(net_params_.nbody_parts);
  for (int body_part_idx = 0; body_part_idx < net_params_.nbody_parts; ++body_part_idx) {
    cv::Mat confidence_map = heatmaps[body_part_idx];
    cv::Mat nms_kernel = NMSRegisterKernel(confidence_map);
    std::vector<KeypointInfo> peaks =
        AccuratePeakPositions(confidence_map, nms_kernel, scaling_factor);
    keypoints.emplace_back(peaks);
  }  // for body parts
  return keypoints;
}

PostprocPose::BodyPairs
PostprocPose::GetBodyPairs(const Heatmaps &heatmaps, const Keypoints &keypoints) {
  const int w = heatmaps[0].cols;
  const int h = heatmaps[0].rows;
  const int paf_offset = net_params_.nbody_parts + 1;  // + 1 for background

  BodyPairs body_pairs;
  for (int pair_idx = 0; pair_idx < net_params_.nbody_part_pairs; ++pair_idx) {
    const int part_idxa = net_params_.body_part_pairs[pair_idx].x;
    const int part_idxb = net_params_.body_part_pairs[pair_idx].y;
    const int paf_idxa = paf_offset + net_params_.paf_indexs[pair_idx].x;
    const int paf_idxb = paf_offset + net_params_.paf_indexs[pair_idx].y;
    const cv::Mat &pafa = heatmaps[paf_idxa];
    const cv::Mat &pafb = heatmaps[paf_idxb];
    const std::vector<KeypointInfo> &peaksa = keypoints[part_idxa];
    const std::vector<KeypointInfo> &peaksb = keypoints[part_idxb];
    const int npeaksa = static_cast<int>(peaksa.size());
    const int npeaksb = static_cast<int>(peaksb.size());
    for (int ai = 0; ai < npeaksa; ++ai) {
      const KeypointInfo &a = peaksa[ai];
      for (int bi = 0; bi < npeaksb; ++bi) {
        const KeypointInfo &b = peaksb[bi];
        const float paf_score = GetPAFScore(a, pafa, b, pafb);
        if (paf_score > 0.f) {
          body_pairs.emplace_back(BodyPairInfo({cv::Point(part_idxa, ai),
                                  cv::Point(part_idxb, bi),
                                  pair_idx,
                                  paf_score,
                                  paf_score + 0.1f * (a.score + b.score)}));
        }
      }  // for peaksb
    }  // for peaksa
  }  // for body pairs
  std::sort(body_pairs.begin(), body_pairs.end(),
      [] (const BodyPairInfo &a, const BodyPairInfo &b) { return a.total_score > b.total_score; });
  return body_pairs;
}

PersonInfos
PostprocPose::GetPersonInfos(const Keypoints &keypoints, const BodyPairs &body_pairs) {
  PersonInfos persons;

  // marks which person the keypoint is assigned to.
  std::vector<std::vector<int>> assigned;
  assigned.reserve(keypoints.size());
  for (const auto &local_keypoints : keypoints) {
    assigned.emplace_back(std::vector<int>(local_keypoints.size(), -1));
  }

  // A & B already assigned to different people, then the two people may be combined into one,
  // another one should be removed. 
  std::set<int, std::greater<int>> persons_to_be_removed;

  for (const auto &body_pair : body_pairs) {
    int part_idxa = net_params_.body_part_pairs[body_pair.body_part_pair_idx].x;
    int part_idxb = net_params_.body_part_pairs[body_pair.body_part_pair_idx].y;
    // which person index keypoint assigned to
    int &assigneda = assigned[body_pair.idxa.x][body_pair.idxa.y];
    int &assignedb = assigned[body_pair.idxb.x][body_pair.idxb.y];
    // keypoints in body pair
    const KeypointInfo &a = keypoints[body_pair.idxa.x][body_pair.idxa.y];
    const KeypointInfo &b = keypoints[body_pair.idxb.x][body_pair.idxb.y];
    // scores
    const float scorea = a.score;
    const float scoreb = b.score;
    const float paf_score = body_pair.score;
    // do keypoints assignment
    if (assigneda < 0 && assignedb < 0) {
      // A & B not assigned yet. new person
      PersonInfo person = {
        std::vector<cv::Point>(net_params_.nbody_parts, kInvalidPos),
        0,   // keypoint counter
        0.f  // score
      };
      person.keypoint_idxs[part_idxa] = body_pair.idxa;
      person.keypoint_idxs[part_idxb] = body_pair.idxb;
      person.nkeypoints = 2;
      person.score = scorea + scoreb + paf_score;
      int person_idx = persons.size();
      assigneda = person_idx;
      assignedb = person_idx;
      persons.emplace_back(person);
    } else if (assigneda >= 0 && assignedb < 0 ||
               assignedb >= 0 && assigneda < 0) {
      // only one of (A, B) assigned. add the other one to person
      int assigned1 = -1, *assigned2 = nullptr, part_idx2 = -1;
      cv::Point part_pos2 = kInvalidPos;
      float score2 = 0.f;
      if (assigneda >= 0) {
        assigned1 = assigneda, assigned2 = &assignedb, part_idx2 = part_idxb,  \
        score2 = scoreb, part_pos2 = body_pair.idxb;
      } else {
        assigned1 = assignedb, assigned2 = &assigneda, part_idx2 = part_idxa,  \
        score2 = scorea, part_pos2 = body_pair.idxa;
      }
      auto &person = persons[assigned1];
      if (person.keypoint_idxs[part_idx2] == kInvalidPos) {
        // if body part not exists.
        person.keypoint_idxs[part_idx2] = part_pos2;
        person.nkeypoints++;
        person.score += score2 + paf_score;
        *assigned2 = assigned1;
      }
      // ignore this limb because the previous one came from a higher score.
    } else if (assigneda == assignedb && assigneda >= 0) {
      // A & B already assigned to same person (circular/redundant PAF). update score
      persons[assigneda].score += paf_score;
    } else if (assigneda >= 0 && assignedb >= 0 && assigneda != assignedb) {
      // A & B already assigned to different people. Merge to the first one if keypoint intersection is null.
      const int assigned1 = assigneda < assignedb ? assigneda : assignedb;
      const int assigned2 = assigneda < assignedb ? assignedb : assigneda;
      auto &person1 = persons[assigned1];
      const auto &person2 = persons[assigned2];
      // merge them if person1 and person2 have the same limb.
      bool merge = true;
      for (int ki = 0; ki < net_params_.nbody_parts; ++ki) {
        if (person1.keypoint_idxs[ki] != kInvalidPos && person2.keypoint_idxs[ki] != kInvalidPos) {
          merge = false;
          break;
        }
      }
      if (merge) {
        // update keypoints and associated
        for (int ki = 0; ki < net_params_.nbody_parts; ++ki) {
          if (person1.keypoint_idxs[ki] == kInvalidPos && person2.keypoint_idxs[ki] != kInvalidPos) {
            person1.keypoint_idxs[ki] = person2.keypoint_idxs[ki];
            assigned[person2.keypoint_idxs[ki].x][person2.keypoint_idxs[ki].y] = assigned1;
          }
        }
        person1.nkeypoints += person2.nkeypoints;
        // update score
        person1.score += paf_score + person2.score;
        // mark person2 for removal
        persons_to_be_removed.insert(assigned2);
      }  // if merge
    }
  }  // for body pairs

  // remove unused people
  // persons_to_be_removed in order from high to low
  for (auto idx : persons_to_be_removed)
    persons.erase(persons.begin() + idx);

  // update score and remove if score less than kSubsetThreshold and number of body pairs less than kMinSubsetCount
  for (auto iter = persons.begin(); iter != persons.end();) {
    float subset_score = iter->score / iter->nkeypoints;
    if (subset_score < kSubsetThreshold || GetNumberBodyPairs(*iter) < kMinSubsetCount) {
      iter = persons.erase(iter);
    } else {
      iter->score /= (net_params_.nbody_parts + net_params_.nbody_part_pairs);
      iter++;
    }
  }
  return persons;
}

cv::Mat
PostprocPose::NMSRegisterKernel(const cv::Mat &confidence_map) {
  const int w = confidence_map.cols;
  const int h = confidence_map.rows;
  cv::Mat kernel = cv::Mat::zeros(h, w, CV_8UC1);
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < h; ++y) {
      const float score = confidence_map.at<float>(y, x);
      if (score <= kNMSThreshold) continue;
      if (1 < x && x < (w - 2) && 1 < y && y < (h - 2)) {
        const float tl = confidence_map.at<float>(y - 1, x - 1);
        const float t  = confidence_map.at<float>(y - 1,     x);
        const float tr = confidence_map.at<float>(y - 1, x + 1);
        const float l  = confidence_map.at<float>(y    , x - 1);
        const float r  = confidence_map.at<float>(y    , x + 1);
        const float bl = confidence_map.at<float>(y + 1, x - 1);
        const float b  = confidence_map.at<float>(y + 1,     x);
        const float br = confidence_map.at<float>(y + 1, x + 1);
        if (score > tl && score > t && score > tr && score > l && score > r &&
            score > bl && score > b && score > br)
          kernel.at<uint8_t>(y, x) = 1;
      } else if (x == 1 || x == (w - 2) || y == 1 || y == (h - 2)) {
        // border
        const float tl = (0 < x && 0 < y)             ? confidence_map.at<float>(y - 1, x - 1) : kNMSThreshold;
        const float t  = (0 < y)                      ? confidence_map.at<float>(y - 1,     x) : kNMSThreshold;
        const float tr = (0 < y && x < (w - 1))       ? confidence_map.at<float>(y - 1, x + 1) : kNMSThreshold;
        const float l  = (0 < x)                      ? confidence_map.at<float>(y    , x - 1) : kNMSThreshold;
        const float r  = (x < (w - 1))                ? confidence_map.at<float>(y    , x + 1) : kNMSThreshold;
        const float bl = (y < (h - 1) && 0 < x)       ? confidence_map.at<float>(y + 1, x - 1) : kNMSThreshold;
        const float b  = (y < (h - 1))                ? confidence_map.at<float>(y + 1,     x) : kNMSThreshold;
        const float br = (x < (w - 1) && y < (h - 1)) ? confidence_map.at<float>(y + 1, x + 1) : kNMSThreshold;
        if (score >= tl && score >= t && score >= tr && score >= l && score >= r &&
            score >= bl && score >= b && score >= br)
          kernel.at<uint8_t>(y, x) = 1;
      }  // else if border
    }  // for y
  }  // for x
  return kernel;
}

std::vector<KeypointInfo>
PostprocPose::AccuratePeakPositions(
    const cv::Mat &confidence_map, const cv::Mat &nms_kernel, const float scaling_factor) {
  const int w = confidence_map.cols;
  const int h = confidence_map.rows;
  const float offset = kNMSOffset * scaling_factor;
  std::vector<KeypointInfo> peaks;
  peaks.reserve(kMaxPeaks);
  for (int y = 0; y < h && peaks.size() < kMaxPeaks; ++y) {
    for (int x = 0; x < w && peaks.size() < kMaxPeaks; ++x) {
      if (nms_kernel.at<uint8_t>(y, x) == 0) continue;
      float x_acc = 0.f, y_acc = 0.f, score_acc = 0.f;
      int left = std::max(0, x - 3), right  = std::min(w - 1, x + 3);
      int top  = std::max(0, y - 3), bottom = std::min(h - 1, y + 3);
      for (int dx = left; dx <= right; ++dx) {
        for (int dy = top; dy <= bottom; ++dy) {
          const float score = confidence_map.at<float>(dy, dx);
          if (score > 0) {
            x_acc += dx * score, y_acc += dy * score;
            score_acc += score;
          }  // if score > 0
        }  // for dy
      }  // for dx
      KeypointInfo info;
      info.pos.x = x_acc / score_acc + offset;
      info.pos.y = y_acc / score_acc + offset;
      info.score = confidence_map.at<float>(y, x);
      peaks.emplace_back(info);
    }  // for y
  }  // for x
  return peaks;
}

float
PostprocPose::GetPAFScore(
    const KeypointInfo &a, const cv::Mat &pafa, const KeypointInfo &b, const cv::Mat &pafb) {
  // L(p(u))
  const int w = pafa.cols;
  const int h = pafa.rows;
  const cv::Point2f distance = b.pos - a.pos;
  const float norm2 = std::sqrt(distance.x * distance.x + distance.y * distance.y);
  if (norm2 <= 1e-6) return -1.f;
  const float max_distance = std::max(std::abs(distance.x), std::abs(distance.y));
  const int nsamples = std::max(5,
      std::min(25, static_cast<int>(std::round(std::sqrt(5 * max_distance)))));
  float x_step = distance.x / nsamples;
  float y_step = distance.y / nsamples;
  const float norma = distance.x / norm2;
  const float normb = distance.y / norm2;

  float score_sum = 0.f;
  int count = 0;
  for (int i = 0; i < nsamples; ++i) {
    cv::Point sample;
    sample.x = std::max(0, std::min(w - 1, static_cast<int>(std::round(a.pos.x + x_step * i))));
    sample.y = std::max(0, std::min(h - 1, static_cast<int>(std::round(a.pos.y + y_step * i))));
    const float score = pafa.at<float>(sample) * norma + pafb.at<float>(sample) * normb;
    if (score >kPAFThreshold) {
      score_sum += score;
      count += 1;
    }  // if > paf threshold
  }  // for samples
  if (1.f * count / nsamples > kMinAboveThreshold) {
    return score_sum / count;
  } else {
    const float threshold = std::sqrt(w * h) / 150.f;
    if (norm2 < threshold) {
      return kNMSThreshold + 1e-6;
    }
  }
  return -1.f;
}

int
PostprocPose::GetNumberBodyPairs(const PersonInfo &person) {
  int count = 0;
  for (auto body_pair : net_params_.body_part_pairs) {
    if (person.keypoint_idxs[body_pair.x] != kInvalidPos &&
        person.keypoint_idxs[body_pair.y] != kInvalidPos)
      count++;
  }
  return count;
}

std::pair<Keypoints, PersonInfos> Postprocess(
      cv::Mat img, float *preds, const float scaling_factor,
      const magicmind::Dims &input_dim, const magicmind::Dims &output_dim) {
  PostprocPose pose;
  return pose.Execute(img, preds, scaling_factor, input_dim, output_dim);
}
