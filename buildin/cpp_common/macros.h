/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Basic collection for common macros
 *************************************************************************/
#ifndef MACROS_H_
#define MACROS_H_
#include <iostream>
#include <string>
#include "logger.h"

#define UNPACK(...) __VA_ARGS__
#define USED_VAR(var) static_cast<void>(var)

#define CHECK_STATUS(status)                \
  do {                                      \
    auto __ret = (status);                  \
    if (__ret != magicmind::Status::OK()) { \
      SLOG(ERROR) << __ret.ToString();      \
      abort();                              \
    }                                       \
  } while (0)

#define CHECK_STATUS_RET(status)            \
  do {                                      \
    auto __ret = (status);                  \
    if (__ret != magicmind::Status::OK()) { \
      return __ret;                         \
    }                                       \
  } while (0)

#define CHECK_VALID(valid)                   \
  do {                                       \
    if (!(valid)) {                          \
      SLOG(ERROR) << #valid " is null or 0"; \
      abort();                               \
    }                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                             \
  do {                                                                             \
    if (a != b) {                                                                  \
      SLOG(ERROR) << #a "(" << a << ") should be equal to " << #b "(" << b << ")"; \
      abort();                                                                     \
    }                                                                              \
  } while (0)

#define CHECK_LE(a, b)                                                                  \
  do {                                                                                  \
    if (a > b) {                                                                        \
      SLOG(ERROR) << #a "(" << a << ") should be less equal to " << #b "(" << b << ")"; \
      abort();                                                                          \
    }                                                                                   \
  } while (0)

#endif  // MACROS_H_
