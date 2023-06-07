/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Wrappers and interfaces for CNRT/CNAPI functions/objects.
 *************************************************************************/
#ifndef DEVICE_H_
#define DEVICE_H_

#include <atomic>
#include <mutex>
#include <vector>
#include <utility>
#include <condition_variable>
#include "cnrt.h"
#include "cndev.h"
#include "cn_api.h"
#include "cnpapi.h"
#include "cnpapi_types.h"
#include "logger.h"
#include "macros.h"
#include "timer.h"

#define CHECK_CNAPI(status)                      \
  do {                                           \
    auto __ret = (status);                       \
    if (__ret != CN_SUCCESS) {                   \
      const char *__str;                         \
      USED_VAR(cnGetErrorString(__ret, &__str)); \
      SLOG(ERROR) << "CNAPI failure: " << __str; \
      abort();                                   \
    }                                            \
  } while (0)

#define CHECK_CNPAPI(status)                          \
  do {                                                \
    auto __ret = status;                              \
    if (__ret != CNPAPI_SUCCESS) {                    \
      const char *__str;                              \
      USED_VAR(cnpapiGetResultString(__ret, &__str)); \
      SLOG(ERROR) << "CNPAPI failure: " << __str;     \
      abort();                                        \
    }                                                 \
  } while (0)

#define CHECK_CNRT(status)                 \
  do {                                     \
    auto __ret = (status);                 \
    if (__ret != cnrtSuccess) {            \
      SLOG(ERROR) << "CNRT ERROR raised."; \
      CNRT_CHECK(__ret);                   \
      abort();                             \
    }                                      \
  } while (0)

#define CHECK_CNRT_RET(status)                                                       \
  do {                                                                               \
    auto __ret = (status);                                                           \
    if (__ret != cnrtSuccess) {                                                      \
      CNRT_CHECK(__ret);                                                             \
      return magicmind::Status(magicmind::error::Code::INTERNAL, #status " failed"); \
    }                                                                                \
  } while (0)
/*
 * class to peek kernel mem usage, should be init before load api and destory after load api
 */
class KernelMemQuery {
 public:
  KernelMemQuery(const std::string &name);
  ~KernelMemQuery();

 private:
  size_t start_ = 0;
  std::string name_;
  KernelMemQuery() = delete;
};
/*
 * Struct to get device static data
 */
struct DeviceInfo {
  DeviceInfo() {
    CHECK_CNRT(cnrtGetDevice(&dev_ordinal_));
    CHECK_CNRT(
        cnrtDeviceGetAttribute(&compute_cap_major_, cnrtAttrComputeCapabilityMajor, dev_ordinal_));
    CHECK_CNRT(
        cnrtDeviceGetAttribute(&compute_cap_minor_, cnrtAttrComputeCapabilityMinor, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&cluster_num_, cnrtAttrClusterCount, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&ipu_clock_rate_, cnrtAttrIpuClockRate, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&mem_clock_rate_, cnrtAttrMemClockRate, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&total_mem_size_, cnrtAttrTotalMemSize, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&mem_bus_width_, cnrtAttrGmemBusWidth, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&max_queue_size_, cnrtAttrQueueSize, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&max_notifier_size_, cnrtAttrNotifierSize, dev_ordinal_));
    CHECK_CNRT(cnrtDeviceGetAttribute(&sparse_computing_support, cnrtAttrSparseComputingSupported,
                                      dev_ordinal_));
  }

  int dev_ordinal_             = -1;
  int compute_cap_major_       = -1;
  int compute_cap_minor_       = -1;
  int cluster_num_             = -1;
  int ipu_clock_rate_          = -1;
  int mem_clock_rate_          = -1;
  int total_mem_size_          = -1;
  int mem_bus_width_           = -1;
  int max_queue_size_          = -1;
  int max_notifier_size_       = -1;
  int sparse_computing_support = -1;
};

std::ostream &operator<<(std::ostream &out, const DeviceInfo &dev_info);
/*
 * Struct to get device utilization data, e.g., power, temperature...
 * We don't give an ostream impl here since usually a group of utilinfo on timeline
 * are valid, but not a single one.
 */
struct DeviceUtilInfo {
  int dev_ordinal_ = -1;
  cndevUtilizationInfo_t util_info_;
  cndevTemperatureInfo_t temp_info_;
  cndevMemoryInfo_t vfmem_info_;
  cndevPowerInfo_t power_info_;
  DeviceUtilInfo(int dev_id) {
    int version           = CNDEV_VERSION_5;
    static cndevRet_t ret = cndevInit(version);
    cndevCheckErrors(ret);
    dev_ordinal_        = dev_id;
    util_info_.version  = version;
    temp_info_.version  = version;
    vfmem_info_.version = version;
    power_info_.version = version;
    cndevCheckErrors(cndevGetDeviceUtilizationInfo(&util_info_, dev_ordinal_));
    cndevCheckErrors(cndevGetTemperatureInfo(&temp_info_, dev_ordinal_));
    cndevCheckErrors(cndevGetPowerInfo(&power_info_, dev_ordinal_));
    cndevCheckErrors(cndevGetMemoryUsage(&vfmem_info_, dev_ordinal_));
  }
};
/*
 * Helpers for get core usage/mem usage/power/temperature
 */
double CoreUtil(const DeviceUtilInfo &t);

double MemUtil(const DeviceUtilInfo &t);

double PowerUtil(const DeviceUtilInfo &t);

double TempUtil(const DeviceUtilInfo &t);
/*
 * Struct to get pmu utilization data, e.g., read/write bytes over time
 * We don't give an ostream impl here since usually a group of utilinfo on timeline
 * are valid, but not a single one.
 * PMUCounter is available in exculsive process.
 */
class PMUCounter {
 public:
  PMUCounter();
  struct PMUUtilInfo {
    int dev_id = -1;
    double dram_read;
    double dram_write;
    double pcie_read;
    double pcie_write;
    double core_read;
    double core_write;
    double alu_cycle;
    double lt_cycle;
  };
  struct PMUData {
    uint64_t now_ = EnvTime::NowMicros();
    int dev_id    = -1;
    std::array<uint64_t, 8> data;
    PMUUtilInfo operator-(const PMUData &d);
  };

  PMUData GetUtil(int dev);

 private:
  std::vector<uint64_t> counter_ids_;
};
/*
 * Helpers for get PMU util
 */
double DRAMRead(const PMUCounter::PMUUtilInfo &t);

double DRAMWrite(const PMUCounter::PMUUtilInfo &t);

double PCIERead(const PMUCounter::PMUUtilInfo &t);

double PCIEWrite(const PMUCounter::PMUUtilInfo &t);

double CoreRead(const PMUCounter::PMUUtilInfo &t);

double CoreWrite(const PMUCounter::PMUUtilInfo &t);

double ALUCycles(const PMUCounter::PMUUtilInfo &t);

double LTCycles(const PMUCounter::PMUUtilInfo &t);
/*
 * Structure for record currently host util:
 * usermode occupancy(%), kernelmode occupancy(%),
 * virtual mem usage(GB), resident set size(GB)
 */
struct HostUtilInfo {
  double user_occ;
  double kernel_occ;
  double vm_usage;
  double res_usage;
};

struct HostUtilData {
  size_t sys_total;
  size_t proc_user;
  size_t proc_kernel;
  double vm_usage;
  double res_usage;
  HostUtilInfo operator-(const HostUtilData &d);
};

HostUtilData GetHostUtil();

double UserOcc(const HostUtilInfo &h);

double KernelOcc(const HostUtilInfo &h);

double VmUsage(const HostUtilInfo &h);

double ResUsage(const HostUtilInfo &h);

bool CheckBindBitmap(int dev_id, const std::vector<int> &cluster_vec);
/*
 * To gain bitmap on dev_id equally among threads 0 1 2...
 * Default bitmap means bind 1 cluster on each thread, e.g.
 * cluster 0 to thread 0, cluster 1 to thread 1, etc.
 */
uint64_t GenBindBitmap(int dev_id, int thread_id, uint64_t bitmap = 0x01);

void BindCluster(int dev_id, uint64_t bitmap);

class Notifier;
/*
 * An interface for MLU queue.
 */
class Queue {
 public:
  Queue();
  explicit Queue(int dev);
  cnrtQueue_t Get() const;
  void Sync() const;
  void Wait(const Notifier *e) const;
  ~Queue();

 private:
  Queue(const Queue &) = delete;
  Queue(Queue &&)      = delete;
  Queue &operator=(const Queue &) = delete;
  Queue &operator=(Queue &&) = delete;
  cnrtQueue_t q_             = nullptr;
};
/*
 * An interface for MLU notifier.
 */
class Notifier {
 public:
  // hardware time tracker only will boost notifier's performance
  explicit Notifier(bool hw_only = false);
  Notifier(int dev, bool hw_only = false);
  ~Notifier();
  cnrtNotifier_t Get() const;
  void PlaceOn(const Queue *queue) const;
  void Wait(const Queue *queue) const;
  void Wait() const;
  // Returns time of host/dev clock in millisecond repectively
  float HostTimeFrom(const Notifier &e) const;
  float DevTimeFrom(const Notifier &e) const;

 private:
  bool hw_only_              = false;
  Notifier(const Notifier &) = delete;
  Notifier(Notifier &&)      = delete;
  Notifier &operator=(const Notifier &) = delete;
  Notifier &operator=(Notifier &&) = delete;
  cnrtNotifier_t n_                = nullptr;
};
/*
 * Host version of Notifier for communicate in threads
 */
class AtomicEvent {
 public:
  AtomicEvent();
  void PlaceOn();
  void Wait(bool reset = true);

 private:
  AtomicEvent(const AtomicEvent &) = delete;
  AtomicEvent(AtomicEvent &&)      = delete;
  AtomicEvent &operator=(const AtomicEvent &) = delete;
  AtomicEvent &operator=(AtomicEvent &&) = delete;
  std::atomic<bool> e_;
  std::mutex mtx_;
  std::condition_variable cv_;
};
#endif  // DEVICE_H_
