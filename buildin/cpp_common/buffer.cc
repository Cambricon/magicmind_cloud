/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: A wrapper for inference I/O buffer/tensors.
 *************************************************************************/
#include <algorithm>
#include "buffer.h"
#include "logger.h"
#include "data.h"
#include "macros.h"

namespace {
static std::map<magicmind::TensorLocation, std::string> kTensorLocationMap{
    {magicmind::TensorLocation::kHost, "kHost"},
    {magicmind::TensorLocation::kMLU, "kMLU"},
    {magicmind::TensorLocation::kRemoteHost, "kRemoteHost"},
    {magicmind::TensorLocation::kRemoteMLU, "kRemoteMLU"}};

std::string TensorLocationEnumToString(magicmind::TensorLocation loc) {
  auto iter = kTensorLocationMap.find(loc);
  if (iter != kTensorLocationMap.end()) {
    return iter->second;
  } else {
    return "INVALID";
  }
}
}  // namespace

void *MLUMalloc(size_t size) {
  void *ptr = nullptr;
  CHECK_CNRT(cnrtMalloc(&ptr, size));
  return ptr;
}

void MLUFree(void *ptr) {
  if (ptr) {
    CHECK_CNRT(cnrtFree(ptr));
  }
}

void *HostMalloc(size_t size) {
  void *ptr = nullptr;
  CHECK_CNRT(cnrtHostMalloc(&ptr, size));
  return ptr;
}

void HostFree(void *ptr) {
  if (ptr) {
    CHECK_CNRT(cnrtFreeHost(ptr));
  }
}

void SetShapes(const std::vector<magicmind::IRTTensor *> &tensors,
               const std::vector<magicmind::Dims> &dims) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->GetDataType() == magicmind::DataType::TENSORLIST) {
      SLOG(ERROR) << "Input tensor list is not supported yet.";
      abort();
    }
    // set first group of input dimension
    CHECK_STATUS(tensors[i]->SetDimensions(dims[i]));
  }
}

Buffers::Buffers(const std::string &name, bool with_host) {
  name_ = name;
  with_host_ = with_host;
}

void Buffers::Init(const std::vector<magicmind::IRTTensor *> &tensors) {
  ori_tensors_ = tensors;
  if (buffers_.size() == 0) {
    for (size_t i = 0; i < ori_tensors_.size(); ++i) {
      if (ori_tensors_[i]->GetDataType() != magicmind::DataType::TENSORLIST) {
        buffers_.push_back(new Buffer(ori_tensors_[i], with_host_));
      }
    }
  }
  ReInit();
}

void Buffers::UnpackTensorListAndUpdateBuffer() {
  ReleaseTensors();
  need_release_ = std::vector<bool>(ori_tensors_.size(), false);
  tensors_ = ori_tensors_;
  for (size_t i = 0, buf_idx = 0; i < tensors_.size();) {
    if (tensors_[i]->GetDataType() == magicmind::DataType::TENSORLIST) {
      // unpack tensor
      std::vector<magicmind::IRTTensor *> unpack_list;
      auto iter_tensors = tensors_.begin() + i;
      CHECK_STATUS(magicmind::GetIRTTensorList(*iter_tensors, &unpack_list));
      tensors_.erase(iter_tensors);
      need_release_.erase(need_release_.begin() + i);
      need_release_.insert(need_release_.begin() + i, unpack_list.size(), true);
      tensors_.insert(tensors_.begin() + i, unpack_list.begin(), unpack_list.end());
      // use buffer
      for (size_t buf_offset = 0; buf_offset < unpack_list.size(); ++buf_offset) {
        if (buf_offset < unpack_map_[i]) {
          // update existed buf
          buffers_[i + buf_offset]->Update(unpack_list[buf_offset]);
        } else {
          // insert new buf if tensorlist is longer than before
          buffers_.insert(buffers_.begin() + i + buf_offset,
                          new Buffer(unpack_list[buf_offset], with_host_));
        }
      }
      if (unpack_list.size() < unpack_map_[i]) {
        for (size_t buf_offset = unpack_list.size(); buf_offset < unpack_map_[i]; ++buf_offset) {
          buffers_[i + buf_offset]->on_use_ = false;
        }
      } else {
        unpack_map_[i] = unpack_list.size();
      }
      buf_idx += unpack_map_[i];
      i += unpack_list.size();
    } else {
      buffers_[buf_idx]->Update(tensors_[i]);
      ++i;
      ++buf_idx;
    }
  }
}

void Buffers::UpdateTensorAddr() {
  for (uint32_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CHECK_STATUS(tensors_[i]->SetData(buffers_[buf_idx]->dev_addr()));
    } else if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kHost) {
      CHECK_STATUS(tensors_[i]->SetData(buffers_[buf_idx]->host_addr()));
    }
  }
}

void Buffers::ReInit() {
  UnpackTensorListAndUpdateBuffer();
  UpdateTensorAddr();
}

void Buffers::FillIn(const std::vector<std::string> &path) {
  if (path.size() != tensors_.size()) {
    SLOG(ERROR) << "Mismatch file num and input num for " << name_ << ".";
    abort();
  }
  ReInit();
  for (size_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    size_t size = tensors_[i]->GetSize();
    if (!ReadDataFromFile(path[i], buffers_[buf_idx]->host_addr(), size)) {
      SLOG(ERROR) << "Read data failed " << name_ << ": " << tensors_[i]->GetName() << ".";
      abort();
    };
  }
}

void Buffers::FillOut(const std::string &path) {
  for (size_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (!WriteDataToFile(path + "output" + std::to_string(i), buffers_[i]->host_addr(),
                         buffers_[i]->current_size_)) {
      SLOG(ERROR) << "Write data failed: output" << std::to_string(i) << ".";
      abort();
    }
  }
}

// memcpy
void Buffers::D2H(Queue *queue) {
  for (uint32_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CHECK_CNRT(cnrtMemcpyAsync(buffers_[buf_idx]->host_addr(), buffers_[buf_idx]->dev_addr(),
                                 tensors_[i]->GetSize(), queue->Get(),
                                 CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
  }
}

void Buffers::H2D(Queue *queue) {
  for (uint32_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CHECK_CNRT(cnrtMemcpyAsync(buffers_[buf_idx]->dev_addr(), buffers_[buf_idx]->host_addr(),
                                 tensors_[i]->GetSize(), queue->Get(),
                                 CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
  }
}

void Buffers::D2H() {
  for (uint32_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CHECK_CNRT(cnrtMemcpy(buffers_[buf_idx]->host_addr(), buffers_[buf_idx]->dev_addr(),
                            tensors_[i]->GetSize(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
  }
}

void Buffers::H2D() {
  for (uint32_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    if (tensors_[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
      CHECK_CNRT(cnrtMemcpy(buffers_[buf_idx]->dev_addr(), buffers_[buf_idx]->host_addr(),
                            tensors_[i]->GetSize(), CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
  }
}

std::vector<magicmind::IRTTensor *> Buffers::OriTensors() const {
  return ori_tensors_;
}

magicmind::IRTTensor *Buffers::operator[](size_t n) const {
  return ori_tensors_[n];
}

magicmind::IRTTensor *Buffers::operator[](const std::string &name) const {
  return magicmind::FindIRTTensorByName(ori_tensors_, name);
}

void Buffers::ClearBuffers() {
  for (auto e_ : buffers_) {
    delete e_;
  }
  buffers_.clear();
}

void Buffers::ReleaseTensors() {
  for (size_t i = 0; i < tensors_.size(); ++i) {
    if (need_release_[i]) {
      tensors_[i]->Destroy();
    }
  }
}

std::string Buffers::DebugString() const {
  std::stringstream ret;
  ret << "\nBuffers Info: " << name_ << "\n";
  ret << "Num: " << tensors_.size() << "\n";
  for (size_t i = 0, buf_idx = 0; i < tensors_.size(); ++i, ++buf_idx) {
    while (!buffers_[buf_idx]->on_use_) {
      ++buf_idx;
    }
    ret << "[" << i << "]: \n";
    ret << "  Name: " << tensors_[i]->GetName() << "\n";
    ret << "  Datatype: " << magicmind::TypeEnumToString(tensors_[i]->GetDataType()) << "\n";
    ret << "  Layout: " << magicmind::LayoutEnumToString(tensors_[i]->GetLayout()) << "\n";
    ret << "  Dim: " << tensors_[i]->GetDimensions() << "\n";
    ret << "  Size: " << tensors_[i]->GetSize() << "\n";
    ret << "  Record Addr: " << tensors_[i]->GetMutableData() << "\n";
    ret << "  TensorLoc: " << TensorLocationEnumToString(tensors_[i]->GetMemoryLocation()) << "\n";
    ret << "  BufferInfo: "
        << "\n";
    ret << buffers_[buf_idx]->DebugString();
  }
  return ret.str();
}

std::vector<void *> Buffers::GetDeviceData() const {
  std::vector<void *> ptrs;
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    while (!buffers_[i]->on_use_) {
      ++i;
    }
    ptrs.push_back(buffers_[i]->dev_addr());
  }
  return ptrs;
}

std::vector<void *> Buffers::GetHostData() const {
  std::vector<void *> ptrs;
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    while (!buffers_[i]->on_use_) {
      ++i;
    }
    ptrs.push_back(buffers_[i]->host_addr());
  }
  return ptrs;
}
