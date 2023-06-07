/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Some functions for file read and data process.
 *************************************************************************/
#ifndef DATA_H_
#define DATA_H_
#include <limits>
#include <queue>
#include <random>
#include <type_traits>
#include "logger.h"
#include "macros.h"
/*
 * LimitsClamped:
 * A helper To convert numeric type from V to T, and clamp the overflow value to its limits.
 */
template <class T, class V, bool = (sizeof(V) > sizeof(T))>
struct LimitsClamped {
  static T From(const V &v) {
    if (v >= V(std::numeric_limits<T>::max())) {
      return std::numeric_limits<T>::max();
    }
    if (v <= V(std::numeric_limits<T>::lowest())) {
      return std::numeric_limits<T>::lowest();
    }
    return T(v);
  }
};

template <class T, class V>
struct LimitsClamped<T, V, false> {
  static T From(const V &v) { return T(v); }
};

template <class T, class V>
T Clamp(const V &v) {
  return LimitsClamped<T, V>::From(v);
}
/*
 * A function to generate len number of uniform distribution int/float nums from begin to end.
 */
template <class T, bool is_integral = std::is_integral<T>::value>
struct RandDist {};

template <typename T>
struct RandDist<T, false> {
  typedef std::uniform_real_distribution<T> Dist;
};

template <typename T>
struct RandDist<T, true> {
  typedef std::uniform_int_distribution<T> Dist;
};

template <class T>
std::vector<T> GenRand(uint64_t len, T begin, T end, unsigned int seed) {
  CHECK_LE(begin, end);
  std::vector<T> ret(len);
  std::default_random_engine eng(seed);
  typename RandDist<T>::Dist dist(begin, end);
  for (size_t idx = 0; idx < len; ++idx) {
    ret[idx] = dist(eng);
  }
  return ret;
}

/*
 * To find min and max element values from src.
 */
template <class T>
std::pair<double, double> GetMinMax(std::vector<T> src) {
  CHECK_VALID(src.size());
  double min = src[0];
  double max = src[0];
  for (int index = 0; index < src.size(); ++index) {
    if (src[index] > max)
      max = static_cast<double>(src[index]);
    if (src[index] < min) {
      min = static_cast<double>(src[index]);
    }
  }
  return std::make_pair(min, max);
}
/*
 * To split string with given delimiter
 */
std::vector<std::string> StringSplit(const std::string &s, const std::string &delimiter);
/*
 * Add "/" tail to file path if needed.
 */
std::string AddLocalPathIfName(const std::string &filepath);
/*
 * Recursively create folder for given path
 */
bool CreateFolder(const std::string &file_path);
/*
 * Return size of file.
 */
size_t FileSize(const std::string &file_path);
/*
 * To read 'size' chars from file_path to ptr. Return a false for failure.
 */
bool ReadDataFromFile(const std::string &file_path, void *ptr, size_t size);
/*
 * To write 'size' chars from file_path to ptr. Return a false for failure.
 */
bool WriteDataToFile(const std::string &file_path, void *ptr, size_t size);
/*
 * To read all lines(ignore empty lines) from file_path, and split them into lines.
 */
bool ReadListFromFile(const std::string &file_path,
                      std::vector<std::string> *lines,
                      std::vector<std::vector<int>> *shapes);
/*
 * To read all lines from file_path, and split them into image tokens and labels. Return a false for
 * failure.
 */
bool ReadLabelFromFile(const std::string &file_path,
                       std::vector<std::string> *images,
                       std::vector<int> *labels);
/*
 * To find top k element indices from src.
 */
template <class T>
std::vector<int> TopK(int k, std::vector<T> src) {
  CHECK_LE(k, src.size());
  std::priority_queue<std::pair<T, int>> q;
  for (int i = 0; i < src.size(); ++i) {
    if (q.size() < k) {
      q.push(std::pair<T, int>(src[i], i));
    } else if (q.top().first < src[i]) {
      q.pop();
      q.push(std::pair<T, int>(src[i], i));
    }
  }
  std::vector<int> ret(k);
  for (int i = 0; i < k; ++i) {
    ret[k - i - 1] = q.top().second;
    q.pop();
  }
  return ret;
}
/*
 * To compute top1/top5 from labels.
 */
template <typename T>
std::pair<int, int> ComputeTop1Top5(std::vector<std::vector<T>> buffers, std::vector<int> labels) {
  int top1    = 0;
  int top5    = 0;
  const int k = 5;
  CHECK_EQ(buffers.size(), labels.size());
  for (uint32_t i = 0; i < labels.size(); ++i) {
    std::vector<int32_t> result = TopK(k, buffers[i]);
    if (labels[i] == result[0]) {
      top1++;
      top5++;
    }
    for (uint32_t res = 1; res < k; ++res) {
      if (labels[i] == result[res]) {
        top5++;
      }
    }
  }
  return std::make_pair(top1, top5);
}

static const double EPSILON = 1e-9;

enum class Diff { Type1 = 1, Type2 = 2, Type3 = 3, Type4 = 4 };
/*
 * To check data has inf or nan.
 */
template <class T>
bool CheckBound(std::vector<T> data) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (std::isinf(float(data[i])) || std::isnan(float(data[i]))) {
      SLOG(WARNING) << "Found " << data[i] << " in boundary checking";
      return true;
    }
  }
  return false;
}

/*
 * Caculate the mse of two data with 4 different type of formulas:
 * - Diff1 = {{\sum {|data_{eval}-data_{base}|}} \over {\sum {| data_{base} |}}}
 * - Diff2 = \sqrt {{\sum {{\left( data_{eval} - data_{base} \right)}^{2}}} \over {\sum
 * {{data_{base}} ^ {2}}}}
 * - Diff3 = \left (diff3_1, diff3_2 \right )
 *   - diff3_1 = \max {{| {data_{eval}}_{i} - {data_{base}}_{i} |} \over {|  {data_{base}} _ {i} |
 * }}
 *   - diff3_2 = \max {| {data_{eval}}_{i} - {data_{base}}_{i} |}
 * - Diff4 = \left (p1, p2 ,n \right ), n stands for total number of unequal data
 *   - p_1 = {{count_1} \over {n}}, count1 stands for number of evaluated data greater than baseline
 *   - p_2 = {{count_2} \over {n}}, count2 stands for number of evaluated data lesser than baseline
 */
template <class T>
std::vector<float> ComputeDiff(std::vector<T> evaluated_data,
                               std::vector<T> baseline_data,
                               Diff type) {
  if (evaluated_data.size() != baseline_data.size()) {
    SLOG(ERROR) << "Data size not match :[" << evaluated_data.size() << "] vs ["
                << baseline_data.size() << "]";
    return {std::numeric_limits<float>::max()};
  }
  CheckBound(evaluated_data);
  CheckBound(baseline_data);
  float thre = 1e-6;
  switch (type) {
    case Diff::Type1: {
      if (evaluated_data.size() == 0)
        return {0};  // support for 0 element case
      float numerator_sum   = 0.0;
      float denominator_sum = 0.0;
      for (size_t i = 0; i < evaluated_data.size(); i++) {
        // if find nan or inf skip for comparing
        if (std::isnan(float(evaluated_data[i])) && std::isnan(float(baseline_data[i]))) {
          SLOG(INFO) << "Diff1 found nan data at " << i << " in both result.";
          continue;
          // if found inf, checkif same inf or -inf, which can compare directly without exception
        } else if (std::isinf(float(evaluated_data[i])) && std::isinf(float(baseline_data[i]))) {
          if (float(evaluated_data[i]) == float(baseline_data[i])) {
            SLOG(INFO) << "Diff1 found inf data at " << i << " in both result.";
            continue;
          } else {
            SLOG(WARNING) << "Diff1 found different inf data, baseline is " << baseline_data[i]
                          << ", evaluated is " << evaluated_data[i];
            return {std::numeric_limits<float>::max()};
          }
        }
        numerator_sum +=
            fabsf(static_cast<float>(evaluated_data[i]) - static_cast<float>(baseline_data[i]));
        denominator_sum += fabsf(static_cast<float>(baseline_data[i]));
      }
      SLOG(INFO) << "Diff1: numerator sum = " << numerator_sum
                 << ", denominator sum = " << denominator_sum;
      if (denominator_sum == 0) {
        denominator_sum += 1e-9;
      }
      return {numerator_sum / denominator_sum};
    }
    case Diff::Type2: {
      if (evaluated_data.size() == 0)
        return {0};  // support for 0 element case
      float numerator_sum   = 0.0;
      float denominator_sum = 0.0;
      for (size_t i = 0; i < evaluated_data.size(); i++) {
        // if find nan or inf skip for comparing
        if (std::isnan(float(evaluated_data[i])) && std::isnan(float(baseline_data[i]))) {
          SLOG(INFO) << "Diff2 found nan data at " << i << " in both result.";
          continue;
          // if found inf, checkif same inf or -inf, which can compare directly without exception
        } else if (std::isinf(float(evaluated_data[i])) && std::isinf(float(baseline_data[i]))) {
          if (float(evaluated_data[i]) == float(baseline_data[i])) {
            SLOG(INFO) << "Diff2 found inf data at " << i << " in both result.";
            continue;
          } else {
            SLOG(WARNING) << "Diff2 found different inf data, baseline is " << baseline_data[i]
                          << ", evaluated is " << evaluated_data[i];
            return {std::numeric_limits<float>::max()};
          }
        }
        float delta =
            fabsf(static_cast<float>(evaluated_data[i]) - static_cast<float>(baseline_data[i]));
        numerator_sum += powf(delta, 2);
        denominator_sum += powf(fabsf(static_cast<float>(baseline_data[i])), 2);
      }
      SLOG(INFO) << "Diff2: numerator sum = " << numerator_sum
                 << ", denominator sum = " << denominator_sum;
      if (denominator_sum == 0) {
        denominator_sum += 1e-9;
      }
      return {sqrtf(numerator_sum / (denominator_sum))};
    }
    case Diff::Type3: {
      if (evaluated_data.size() == 0)
        return {0, 0};  // support for 0 element case
      float diff_3_1 = 0.0;
      float diff_3_2 = 0.0;
      for (size_t i = 0; i < evaluated_data.size(); i++) {
        // if find nan or inf skip for comparing
        if (std::isnan(float(evaluated_data[i])) && std::isnan(float(baseline_data[i]))) {
          SLOG(INFO) << "Diff3 found nan data at " << i << " in both result.";
          continue;
          // if found inf, checkif same inf or -inf, which can compare directly without exception
        } else if (std::isinf(float(evaluated_data[i])) && std::isinf(float(baseline_data[i]))) {
          if (float(evaluated_data[i]) == float(baseline_data[i])) {
            SLOG(INFO) << "Diff3 found inf data at " << i << " in both result.";
            continue;
          } else {
            SLOG(WARNING) << "Diff3 found different inf data, baseline is " << baseline_data[i]
                          << ", evaluated is " << evaluated_data[i];
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
          }
        }
        float numerator =
            fabs(static_cast<float>(evaluated_data[i]) - static_cast<float>(baseline_data[i]));
        float ratio = numerator / (fabsf(static_cast<float>(baseline_data[i])));
        diff_3_1 =
            ((ratio > diff_3_1) && (fabsf((float)baseline_data[i]) > thre)) ? ratio : diff_3_1;
        diff_3_2 = ((numerator > diff_3_2) && (fabsf((float)baseline_data[i]) <= thre)) ? numerator
                                                                                        : diff_3_2;
      }
      return {diff_3_1, diff_3_2};
    }
    case Diff::Type4: {
      if (evaluated_data.size() == 0)
        return {0, 0, 0};  // support for 0 element case
      int count_1    = 0;
      int count_2    = 0;
      int total_data = evaluated_data.size();
      for (size_t i = 0; i < evaluated_data.size(); i++) {
        // if find nan or inf skip for comparing
        if (std::isnan(float(evaluated_data[i])) && std::isnan(float(baseline_data[i]))) {
          total_data -= 1;
          SLOG(INFO) << "Diff4 found nan data at " << i << " in both result.";
          continue;
          // if found inf, checkif same inf or -inf, which can compare directly without exception
        } else if (std::isinf(float(evaluated_data[i])) && std::isinf(float(baseline_data[i]))) {
          if (float(evaluated_data[i]) == float(baseline_data[i])) {
            total_data -= 1;
            SLOG(INFO) << "Diff4 found inf data at " << i << " in both result.";
            continue;
          } else {
            SLOG(WARNING) << "Diff4 found different inf data, baseline is " << baseline_data[i]
                          << ", evaluated is " << evaluated_data[i];
            return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    static_cast<float>(evaluated_data.size())};
          }
        }
        if ((float)evaluated_data[i] > (float)baseline_data[i]) {
          count_1++;
        } else if ((float)evaluated_data[i] < (float)baseline_data[i]) {
          count_2++;
        } else {
          // when evaluated data equals baseline data, does not count in diff4 total num.
          total_data -= 1;
        }
      }
      // when full output total number >= 100, count diff4, or may random fail.
      if (evaluated_data.size() < 100) {
        SLOG(INFO) << "Diff4 {" << float(count_1) / total_data << " " << float(count_2) / total_data
                   << "@ " << total_data
                   << "} should not count because total number is less than 100";
      }
      if (total_data == 0) {
        SLOG(INFO) << "Diff4 return {0, 0}, because different total data is 0.";
        return {0, 0, static_cast<float>(total_data)};
      } else {
        return {float(count_1) / total_data, float(count_2) / total_data,
                static_cast<float>(total_data)};
      }
    }
  }
  SLOG(ERROR) << "Bad type for diff.";
  abort();
}

#endif  // DATA_H_
