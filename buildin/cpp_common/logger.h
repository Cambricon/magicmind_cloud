/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: A simple implement for a light-weight logger
 *************************************************************************/
#ifndef LOGGER_H_
#define LOGGER_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <atomic>
#include <limits>
#include <memory>
#include <sstream>

/*
 * A logger which supports 3 kinds of severities: info/warning/error.
 * To use it, call macro SLOG(INFO/WARNING/ERROR) << in corresponding line.
 * LOG will be print as follow:
 * y-m-d h:m:s.micro: INFO/WARNING/ERROR:  filepath:line] Your Content
 */

const int INFO           = 0;
const int WARNING        = 1;
const int ERROR          = 2;
const int NUM_SEVERITIES = 3;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char *fname, int line, int severity)
      : fname_(fname), line_(line), severity_(severity) {}
  ~LogMessage() override { GenerateLogMessage(); }

 public:
  void GenerateLogMessage();

 private:
  const char *fname_;
  int line_;
  int severity_;
};

#define _SAMPLE_LOG_INFO LogMessage(__FILE__, __LINE__, INFO)
#define _SAMPLE_LOG_WARNING LogMessage(__FILE__, __LINE__, WARNING)
#define _SAMPLE_LOG_ERROR LogMessage(__FILE__, __LINE__, ERROR)

/*
 * SLOG stands for "sample log", to avoid duplicated defination
 * for macro "LOG".
 */
#define SLOG(severity) _SAMPLE_LOG_##severity

namespace std {
template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i < (vec.size() - 1)) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

template <class T1, class T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p) {
  os << p.first << ": " << p.second;
  return os;
}

template <class T1, class T2>
inline std::ostream &operator<<(std::ostream &os, const std::map<T1, T2> &map) {
  os << "[";
  size_t count = 0;
  for (auto e_ : map) {
    os << e_;
    if (count < (map.size() - 1)) {
      os << ", ";
    }
    ++count;
  }
  os << "]";
  return os;
}

}  // namespace std

#endif  // LOGGER_H_
