/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: A wrapper for inference I/O buffers/tensors.
 *************************************************************************/
#ifndef BUFFER_H_
#define BUFFER_H_
#include <map>
#include <functional>
#include <string>
#include "cnrt.h"
#include "cn_api.h"
#include "mm_runtime.h"
#include "device.h"
/*
 * Functions wrapped for malloc and free.
 */
void *MLUMalloc(size_t size);

void MLUFree(void *ptr);

void *HostMalloc(size_t size);

void HostFree(void *ptr);
/*
 * Function wrapped for set shapes for tensors
 */
void SetShapes(const std::vector<magicmind::IRTTensor *> &tensors,
               const std::vector<magicmind::Dims> &dims);
/*
 * A class creates input/output mlu/cpu buffers according to input/output IRTTensors,
 * and support memcpy sync/async between mlu and cpu buffers.
 * Buffers will try to reuse same input/output address buffer if possible.
 */
class Buffers {
 public:
  Buffers(const std::string &name, bool with_host = true);

  ~Buffers() {
    ReleaseTensors();
    ClearBuffers();
  }
  /*
   * To init buffers with tensors.
   * Recorded buffers will be free and remalloc for new tensors.
   * TensorList will be unpack to Tensors after init and reinit.
   * with_host mean to malloc with host mem, disable when host mem is not in use(e.g.disable data
   * cpy)
   */
  void Init(const std::vector<magicmind::IRTTensor *> &tensors);
  /*
   * To reinit buffers with recorded tensors.
   * Recorded buffers will be free and remalloc if size is not enough.
   * TensorList will be unpack to Tensors after init and reinit.
   */
  void ReInit();
  /*
   * To fill buffers with batches of data.
   * Recorded buffers will be free and remalloc if size is not enough.
   */
  void FillIn(const std::vector<std::string> &path);
  /*
   * To write buffers out to path/output[idx].
   */
  void FillOut(const std::string &path);

  void D2H(Queue *queue);

  void H2D(Queue *queue);

  void D2H();

  void H2D();

  std::vector<magicmind::IRTTensor *> OriTensors() const;

  magicmind::IRTTensor *operator[](size_t n) const;

  magicmind::IRTTensor *operator[](const std::string &name) const;

  std::string DebugString() const;

  std::vector<void *> GetDeviceData() const;

  std::vector<void *> GetHostData() const;

 private:
  Buffers(const Buffers &) = delete;
  Buffers(Buffers &&)      = delete;
  Buffers &operator=(const Buffers &) = delete;
  Buffers &operator=(Buffers &&) = delete;
  /*
   * A function to expand tensorlist to flat tensors for further memory management.
   */
  void UnpackTensorListAndUpdateBuffer();
  void TrackingAllocator(int index, size_t size, bool host);
  void UpdateTensorAddr();
  void ClearBuffers();
  void ReleaseTensors();

 private:
  std::string name_;
  bool with_host_ = true;
  struct Buffer {
    Buffer()               = delete;
    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&)      = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) = delete;
    Buffer(magicmind::IRTTensor *t, bool malloc_host) {
      malloc_host_ = malloc_host;
      loc_         = t->GetMemoryLocation();
      if ((loc_ != magicmind::TensorLocation::kMLU) && (loc_ != magicmind::TensorLocation::kHost)) {
        SLOG(ERROR) << "Internal error: Unsupport memloc type in buffer " << t->GetName() << ".";
        abort();
      }
      if (t->GetMutableData()) {
        // there is data which already exist in tensor
        malloc_by_us_ = false;
      }
      Update(t);
    }
    // the key point is to make sure tensor is malloc by us or not.
    void Update(magicmind::IRTTensor *t) {
      on_use_   = true;
      auto size = t->GetSize();
      if (!malloc_by_us_) {
        // assign addr
        if (loc_ == magicmind::TensorLocation::kHost) {
          // host addr belongs to runtime, only happens when model is mutable
          host_addrs_.push_back(t->GetMutableData());
        } else {
          // mlu addr belongs to runtime, only happens when model is mutable
          dev_addrs_.push_back(t->GetMutableData());
        }
      }
      // remalloc part
      bool remalloc = false;
      if (size > current_size_) {
        current_size_ = size;
        remalloc      = true;
      }
      if (remalloc) {
        if (loc_ == magicmind::TensorLocation::kMLU) {
          // malloc for mlu addr's host cpy
          if (malloc_host_) {
            host_addrs_.push_back(HostMalloc(size));
            total_host_size_ += size;
          }
          if (malloc_by_us_) {
            // malloc for mlu addr which should be malloc by us
            // will not do mlu malloc for pure host tensor
            dev_addrs_.push_back(MLUMalloc(size));
            total_dev_size_ += size;
          }
        } else {
          // host ptr need no mlu addr
          if (malloc_by_us_) {
            // malloc for host addr which should be malloc by us
            host_addrs_.push_back(HostMalloc(size));
            total_host_size_ += size;
          }
        }
      }
    }
    ~Buffer() {
      if (malloc_by_us_) {
        for (auto addr : host_addrs_) {
          HostFree(addr);
        }
        for (auto addr : dev_addrs_) {
          MLUFree(addr);
        }
      } else {
        if (loc_ == magicmind::TensorLocation::kMLU) {
          for (auto addr : host_addrs_) {
            HostFree(addr);
          }
        }
      }
    }
    // return current using host addr
    void *host_addr() const { return host_addrs_.size() > 0 ? host_addrs_.back() : nullptr; }
    // return current using host addr
    void *dev_addr() const { return dev_addrs_.size() > 0 ? dev_addrs_.back() : nullptr; }
    std::string DebugString() const {
      std::stringstream ret;
      ret << "    Malloc by us: " << malloc_by_us_ << "\n";
      ret << "    Current Host Addr: " << host_addr() << "\n";
      ret << "    Total Host Size: " << total_host_size_ << "\n";
      ret << "    Host Addr depth: " << host_addrs_.size() << "\n";
      ret << "    Current Dev Addr: " << dev_addr() << "\n";
      ret << "    Total Dev Size: " << total_dev_size_ << "\n";
      ret << "    Dev Addr depth: " << dev_addrs_.size() << "\n";
      ret << "    Current Size: " << current_size_ << "\n";
      return ret.str();
    }
    bool malloc_by_us_ = true;
    bool on_use_       = true;
    bool malloc_host_  = true;
    magicmind::TensorLocation loc_;
    std::vector<void *> host_addrs_;
    std::vector<void *> dev_addrs_;
    size_t current_size_    = 0;
    size_t total_host_size_ = 0;
    size_t total_dev_size_  = 0;
  };
  std::vector<Buffer *> buffers_               = {};
  std::vector<magicmind::IRTTensor *> tensors_ = {};
  std::vector<bool> need_release_              = {};
  // To record tensorlist length, oritensor index : buffer length
  std::map<size_t, size_t> unpack_map_             = {};
  std::vector<magicmind::IRTTensor *> ori_tensors_ = {};
};

#endif  // BUFFER_H_
