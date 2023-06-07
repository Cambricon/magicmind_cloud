/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: A simple implement for a light-weight logger
 *************************************************************************/
#include <stdint.h>
#include "logger.h"
#include "timer.h"

void LogMessage::GenerateLogMessage() {
  uint64_t now_micros = EnvTime::NowMicros();
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  char severity_buffer[4][10] = {"INFO", "WARNING", "ERROR"};
  fprintf(stderr, "%s.%06d: %s: %s:%d] %s\n", EnvTime::CurrentTime().c_str(), micros_remainder,
          severity_buffer[severity_], fname_, line_, str().c_str());
}
