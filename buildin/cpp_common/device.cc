/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Wrappers and interfaces for CNRT/CNAPI functions/objects.
 *************************************************************************/
#include <iomanip>
#include <functional>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include "device.h"

KernelMemQuery::KernelMemQuery(const std::string &name) : name_(name) {
  CHECK_CNRT(cnrtDeviceQueryKernelMemoryUsage(&start_));
}

KernelMemQuery::~KernelMemQuery() {
  size_t end = 0;
  CHECK_CNRT(cnrtDeviceQueryKernelMemoryUsage(&end));
  if ((end - start_) > 0) {
    SLOG(INFO) << "Kernel mem occ during " << name_ << " is " << end - start_;
  } else {
    SLOG(INFO) << "No kernel mem loading occured during " << name_
               << ", this may because of a cropped lib or lib is preloaded.";
  }
}

std::ostream &operator<<(std::ostream &out, const DeviceInfo &dev_info) {
  out << std::endl;
  out << "=================== "
      << "Device Information" << std::endl;
  out << std::setw(30) << std::left << "Device ID: " << dev_info.dev_ordinal_ << std::endl;
  out << std::setw(30) << std::left << "Compute Capability: " << dev_info.compute_cap_major_ << "."
      << dev_info.compute_cap_minor_ << std::endl;
  out << std::setw(30) << std::left << "Cluster Number: " << dev_info.cluster_num_ << std::endl;
  out << std::setw(30) << std::left << "MLU Core Clock Rate: " << dev_info.ipu_clock_rate_ / 1e6
      << " (GHz)" << std::endl;
  out << std::setw(30) << std::left << "Total Memory Size: " << dev_info.total_mem_size_ << " (MB)"
      << std::endl;
  out << std::setw(30) << std::left << "Memory Bus Width: " << dev_info.mem_bus_width_ << " (bits)"
      << std::endl;
  out << std::setw(30) << std::left << "Memory Clock Rate: " << dev_info.mem_clock_rate_ / 1e6
      << " (GHz)" << std::endl;
  out << std::setw(30) << std::left << "Maximum Queue Size: " << dev_info.max_queue_size_
      << std::endl;
  out << std::setw(30) << std::left << "Maximum Notifer Size: " << dev_info.max_notifier_size_
      << std::endl;
  out << std::setw(30) << std::left
      << "Sparse Computing Support: " << dev_info.sparse_computing_support << std::endl;
  out << std::setw(30) << std::left
      << "Host Memory Map Support: " << dev_info.sparse_computing_support;
  return out;
}

double CoreUtil(const DeviceUtilInfo &t) {
  return t.util_info_.averageCoreUtilization;
}

double MemUtil(const DeviceUtilInfo &t) {
  return t.vfmem_info_.physicalMemoryUsed;
}

double PowerUtil(const DeviceUtilInfo &t) {
  return t.power_info_.usage;
}

double TempUtil(const DeviceUtilInfo &t) {
#ifdef __aarch64__
  return t.temp_info_.board;
#else
  return t.temp_info_.chip;
#endif
}

HostUtilData GetHostUtil() {
  std::ifstream proc_stat;
  HostUtilData ret;
  std::string none;
  unsigned long vsize = 0;
  long rss = 0;
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
  ret.sys_total = 0;
  {
    std::ifstream proc_whole("/proc/stat", std::ios_base::in);
    proc_whole >> none;
    for (size_t time = 0; proc_whole >> time; ret.sys_total += time)
      ;
    proc_whole.close();
  }
  {
    std::ifstream proc_self("/proc/self/stat", std::ios_base::in);
    // 0 - 12: pid exname state ppid pgrp session
    // tty tgid flags minflt cminflt majflt cmagflt
    for (int counter = 0; counter < 13; ++counter) {
      proc_self >> none;
    }
    // 13 - 14: utime stime
    proc_self >> ret.proc_user;
    proc_self >> ret.proc_kernel;
    // 15 - 21: cutime cstime priority nice numthreads
    // itrealvalue starttime
    for (int counter = 0; counter < 7; ++counter) {
      proc_self >> none;
    }
    // 22 - 23: vsize rss
    proc_self >> vsize;
    proc_self >> rss;
    proc_self.close();
  }
  ret.vm_usage = double(vsize / 1024.0) / 1024.0 / 1024.0;
  ret.res_usage = double(rss * page_size_kb) / 1024.0 / 1024.0;
  return ret;
}

HostUtilInfo HostUtilData::operator-(const HostUtilData &d) {
  HostUtilInfo ret;
  double sys = sys_total - d.sys_total;
  if (sys == 0) {
    ret.user_occ = 0;
    ret.kernel_occ = 0;
  } else {
    ret.user_occ = 100 * double(proc_user - d.proc_user) / sys;
    ret.kernel_occ = 100 * double(proc_kernel - d.proc_kernel) / sys;
  }
  ret.vm_usage = d.vm_usage;
  ret.res_usage = d.res_usage;
  return ret;
}

double UserOcc(const HostUtilInfo &h) {
  return h.user_occ;
}

double KernelOcc(const HostUtilInfo &h) {
  return h.kernel_occ;
}

double VmUsage(const HostUtilInfo &h) {
  return h.vm_usage;
}

double ResUsage(const HostUtilInfo &h) {
  return h.res_usage;
}

bool CheckBindBitmap(int dev_id, const std::vector<int> &cluster_vec) {
  int cluster_num = 0;
  CHECK_CNRT(cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, dev_id));
  for (auto index : cluster_vec) {
    if (index > cluster_num) {
      SLOG(ERROR) << "Unable to bind cluster: " << index
                  << ", which is bigger than the cluster number supported by the device: "
                  << cluster_num << ".";
      return false;
    }
  }
  return true;
}

uint64_t GenBindBitmap(int dev_id, int thread_id, uint64_t bitmap) {
  int cluster_num = 0;
  CHECK_CNRT(cnrtDeviceGetAttribute(&cluster_num, cnrtAttrClusterCount, dev_id));
  return bitmap << (thread_id % cluster_num);
}

void BindCluster(int dev_id, uint64_t bitmap) {
  // Set device id and get max cluster num
  CHECK_CNRT(cnrtSetDevice(dev_id));

  // Set ctx
  CNcontext ctx;
  CHECK_CNAPI(cnCtxGetCurrent(&ctx));

  CNctxConfigParam param;
  CHECK_CNAPI(cnGetCtxConfigParam(ctx, CN_CTX_CONFIG_UNION_LIMIT, &param));
  // Uses block task when the hardware only supports block task,
  // otherwise uses union task.
  if (param.unionLimit != CN_KERNEL_CLASS_BLOCK) {
    param.unionLimit = CN_KERNEL_CLASS_UNION;
  }
  CHECK_CNAPI(cnSetCtxConfigParam(ctx, CN_CTX_CONFIG_UNION_LIMIT, &param));
  param.visibleCluster = bitmap;
  CHECK_CNAPI(cnSetCtxConfigParam(ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER, &param));
}

Queue::Queue() {
  CHECK_CNRT(cnrtQueueCreate(&q_));
}

Queue::Queue(int dev) {
  int ordinal = -1;
  CHECK_CNRT(cnrtSetDevice(dev));
  CHECK_CNRT(cnrtGetDevice(&ordinal));
  CHECK_EQ(ordinal, dev);
  CHECK_CNRT(cnrtQueueCreate(&q_));
}

Queue::~Queue() {
  CHECK_CNRT(cnrtQueueDestroy(q_));
}

cnrtQueue_t Queue::Get() const {
  return q_;
}

void Queue::Sync() const {
  CHECK_CNRT(cnrtQueueSync(q_));
}

void Queue::Wait(const Notifier *e) const {
  CHECK_CNRT(cnrtQueueWaitNotifier(e->Get(), q_, 0));
}

Notifier::Notifier(bool hw_only) : hw_only_(hw_only) {
  if (hw_only) {
    CHECK_CNRT(cnrtNotifierCreateWithFlags(&n_, 0x02));
  } else {
    CHECK_CNRT(cnrtNotifierCreate(&n_));
  }
}

Notifier::Notifier(int dev, bool hw_only) : hw_only_(hw_only) {
  int ordinal = -1;
  CHECK_CNRT(cnrtSetDevice(dev));
  CHECK_CNRT(cnrtGetDevice(&ordinal));
  CHECK_EQ(ordinal, dev);
  if (hw_only) {
    CHECK_CNRT(cnrtNotifierCreateWithFlags(&n_, 0x02));
  } else {
    CHECK_CNRT(cnrtNotifierCreate(&n_));
  }
}

Notifier::~Notifier() {
  CHECK_CNRT(cnrtNotifierDestroy(n_));
}

cnrtNotifier_t Notifier::Get() const {
  return n_;
}

void Notifier::PlaceOn(const Queue *queue) const {
  CHECK_CNRT(cnrtPlaceNotifier(n_, queue->Get()));
}

void Notifier::Wait() const {
  CHECK_CNRT(cnrtWaitNotifier(n_));
}

void Notifier::Wait(const Queue *queue) const {
  queue->Wait(this);
}

float Notifier::HostTimeFrom(const Notifier &e) const {
  CHECK_VALID(!hw_only_);
  float ret = 0;
  CHECK_CNRT(cnrtNotifierElapsedTime(e.Get(), Get(), &ret));
  return ret;
}

float Notifier::DevTimeFrom(const Notifier &e) const {
  float ret = 0;
  CHECK_CNRT(cnrtNotifierDuration(e.Get(), Get(), &ret));
  return ret / 1000;
}

AtomicEvent::AtomicEvent() {
  e_ = false;
}

void AtomicEvent::PlaceOn() {
  std::unique_lock<std::mutex> lk(mtx_);
  e_.store(true);
  cv_.notify_all();
}

void AtomicEvent::Wait(bool reset) {
  std::unique_lock<std::mutex> lk(mtx_);
  cv_.wait(lk, [this]() { return e_.load(); });
  if (reset) {
    e_.store(false);
  }
}
