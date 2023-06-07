/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: An timer implement for system time and program benchmark.
 *************************************************************************/
#include <string>
#include "logger.h"
#include "timer.h"

uint64_t EnvTime::NowNanos(clockid_t cid) {
  struct timespec ts;
  clock_gettime(cid, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
}

uint64_t EnvTime::NowMicros(clockid_t cid) {
  return NowNanos(cid) / kMicrosToNanos;
}

std::string EnvTime::CurrentTime() {
  uint64_t now_micros = EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  char time_buffer[TimeBufferSize];
  struct tm time_tm_buf;
  strftime(time_buffer, TimeBufferSize, "%Y-%m-%d %H:%M:%S",
           localtime_r(&now_seconds, &time_tm_buf));
  return std::string(time_buffer);
}

TimeCollapse::TimeCollapse(const std::string &name) : name_(name) {
  start_ = EnvTime::NowMicros(CLOCK_MONOTONIC);
}

TimeCollapse::~TimeCollapse() {
  uint64_t end = EnvTime::NowMicros(CLOCK_MONOTONIC);
  uint64_t dur = end - start_;
  SLOG(INFO) << name_ << " time is " << float(dur) / 1000 << " ms.";
}
