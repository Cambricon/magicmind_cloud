/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: An timer implement for system time and program benchmark.
 *************************************************************************/
#ifndef TIMER_H_
#define TIMER_H_
#include <time.h>
#define TimeBufferSize 30
/*
 * A struct using clock_gettime() for system time and steady time.
 */
struct EnvTime {
  static constexpr uint64_t kMicrosToNanos  = 1000ULL;
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;
  static uint64_t NowNanos(clockid_t cid = CLOCK_REALTIME);
  static uint64_t NowMicros(clockid_t cid = CLOCK_REALTIME);
  static std::string CurrentTime();
};

/*
 * A class to calculate time from its construction to destruction.
 */
class TimeCollapse {
 public:
  TimeCollapse(const std::string &name);
  ~TimeCollapse();

 private:
  uint64_t start_;
  std::string name_;
  TimeCollapse() = delete;
};

#endif  // TIMER_H_
