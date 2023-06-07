#include "model_runner.h"
#include <iomanip>
std::string PrintEngineInfo(magicmind::IEngine *engine,
                            std::vector<magicmind::Dims> max_input_dims,
                            float &meminfo) {
  std::stringstream ss;
  ss.precision(3);
  uint64_t const_size = 0;
  uint64_t ctx_size = 0;
  MM_CHECK(engine->QueryConstDataSize(&const_size));
  ss << "Constdata Size:" << std::setprecision(2) << const_size * 1.0 / 1024 / 1024 << " (MB)   ";
  auto ret = engine->QueryContextMaxWorkspaceSize(max_input_dims, &ctx_size);
  if (ret.ok()) {
    ss << "ContextMaxworkspace Size:" << std::setprecision(2)
       << std::to_string(ctx_size * 1.0 / 1024 / 1024) << " (MB)";
  } else {
    ss << "ContextMaxworkspace Size: UNAVAILABLE.\n";
  }
  meminfo += const_size * 1.0 / 1024 / 1024;
  meminfo += ctx_size * 1.0 / 1024 / 1024;
  return ss.str();
}

ModelRunner::ModelRunner(int dev_id, std::string model_path) {
  dev_id_ = dev_id;
  model_path_ = model_path;
}

bool ModelRunner::Init(int max_batch) {
  CNRT_CHECK(cnrtSetDevice(dev_id_));
  queue_ = new Queue();
  model_ = magicmind::CreateIModel();
  MM_CHECK(model_->DeserializeFromFile(model_path_.c_str()));
  //[-1,h,w,c]
  auto input_dims = model_->GetInputDimensions();
  engine_ = model_->CreateIEngine();
  PTR_CHECK(engine_);
  context_ = engine_->CreateIContext();
  PTR_CHECK(context_);
  input_buffer_ = new Buffers("Input");
  output_buffer_ = new Buffers("Output");
  MM_CHECK(context_->CreateInputTensors(&input_tensors_));
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    auto v = input_dims[i].GetDims();
    v[0] = max_batch;
    MM_CHECK(input_tensors_[i]->SetDimensions(magicmind::Dims(v)));
  }
  input_buffer_->Init(input_tensors_);
  return true;
}

bool ModelRunner::Init(std::vector<std::vector<int64_t>> &max_dims) {
  CNRT_CHECK(cnrtSetDevice(dev_id_));
  queue_ = new Queue();
  model_ = magicmind::CreateIModel();
  MM_CHECK(model_->DeserializeFromFile(model_path_.c_str()));
  //[-1,-1..-1]
  auto input_dims = model_->GetInputDimensions();
  engine_ = model_->CreateIEngine();
  PTR_CHECK(engine_);
  context_ = engine_->CreateIContext();
  PTR_CHECK(context_);
  input_buffer_ = new Buffers("Input");
  output_buffer_ = new Buffers("Output");
  MM_CHECK(context_->CreateInputTensors(&input_tensors_));
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    MM_CHECK(input_tensors_[i]->SetDimensions(magicmind::Dims(max_dims[i])));
  }
  input_buffer_->Init(input_tensors_);
  return true;
}

std::vector<std::vector<int64_t>> ModelRunner::GetInputDims() {
  std::vector<std::vector<int64_t>> dims;
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    dims.push_back(input_tensors_[i]->GetDimensions().GetDims());
  }
  return dims;
}

std::vector<std::vector<int64_t>> ModelRunner::GetOutputDims() {
  std::vector<std::vector<int64_t>> dims;
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    dims.push_back(output_tensors_[i]->GetDimensions().GetDims());
  }
  return dims;
}

std::vector<int64_t> ModelRunner::GetInputSizes() {
  std::vector<int64_t> sizes;
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    sizes.push_back(input_tensors_[i]->GetSize());
  }
  return sizes;
}

std::vector<int64_t> ModelRunner::GetOutputSizes() {
  std::vector<int64_t> sizes;
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    sizes.push_back(output_tensors_[i]->GetSize());
  }
  return sizes;
}

void ModelRunner::Run(int n, Queue *queue) {
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    auto v = input_tensors_[i]->GetDimensions().GetDims();
    v[0] = n;
    MM_CHECK(input_tensors_[i]->SetDimensions(magicmind::Dims(v)));
  }
  MM_CHECK(context_->Enqueue(input_tensors_, &output_tensors_, queue->Get()));
  output_buffer_->Init(output_tensors_);
}

void ModelRunner::Run(int n) {
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    auto v = input_tensors_[i]->GetDimensions().GetDims();
    v[0] = n;
    MM_CHECK(input_tensors_[i]->SetDimensions(magicmind::Dims(v)));
  }
  MM_CHECK(context_->Enqueue(input_tensors_, &output_tensors_, queue_->Get()));
  output_buffer_->Init(output_tensors_);
  queue_->Sync();
}

std::vector<magicmind::DataType> ModelRunner::GetInputDataTypes() {
  std::vector<magicmind::DataType> types;
  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    types.push_back(input_tensors_[i]->GetDataType());
  }
  return types;
}

std::vector<magicmind::DataType> ModelRunner::GetOutputDataTypes() {
  std::vector<magicmind::DataType> types;
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    types.push_back(output_tensors_[i]->GetDataType());
  }
  return types;
}

std::vector<void *> ModelRunner::GetHostOutputData() {
  return output_buffer_->GetHostData();
}

std::vector<void *> ModelRunner::GetHostInputData() {
  return input_buffer_->GetHostData();
}

void ModelRunner::PrintMemInfo(float &meminfo) {
  SLOG(INFO) << PrintEngineInfo(engine_, {input_tensors_[0]->GetDimensions()}, meminfo);
  SLOG(INFO) << "input buffer:\n" << input_buffer_->DebugString();
  SLOG(INFO) << "output buffer:\n" << output_buffer_->DebugString();
}

void ModelRunner::Destroy() {
  if (input_buffer_) {
    delete input_buffer_;
    input_buffer_ = nullptr;
  }
  if (output_buffer_) {
    delete output_buffer_;
    output_buffer_ = nullptr;
  }

  for (uint32_t i = 0; i < input_tensors_.size(); ++i) {
    input_tensors_[i]->Destroy();
  }
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    output_tensors_[i]->Destroy();
  }

  if (context_) {
    context_->Destroy();
    context_ = nullptr;
  }

  if (engine_) {
    engine_->Destroy();
    engine_ = nullptr;
  }
  if (model_) {
    model_->Destroy();
    model_ = nullptr;
  }
  delete queue_;
  delete this;
}
