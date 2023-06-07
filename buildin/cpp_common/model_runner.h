#ifndef MODEL_RUNNER_H_
#define MODEL_RUNNER_H_

#include "buffer.h"

#define MM_CHECK(status)                                  \
  do {                                                    \
    if ((status) != magicmind::Status::OK()) {            \
      std::cout << "mm failure: " << status << std::endl; \
      abort();                                            \
    }                                                     \
  } while (0)
#define PTR_CHECK(ptr)                         \
  do {                                         \
    if (ptr == nullptr) {                      \
      std::cout << "mm failure " << std::endl; \
      abort();                                 \
    }                                          \
  } while (0)
#define MM_CNAPI_CHECK(status)                                   \
  do {                                                           \
    auto __ret = status;                                         \
    if ((__ret) != CN_SUCCESS) {                                 \
      std::cout << "mm cn_api failure: " << status << std::endl; \
      abort();                                                   \
    }                                                            \
  } while (0)

class ModelRunner {
 public:
  ModelRunner(int dev_id, std::string model_path);
  bool Init(int batch);
  bool Init(std::vector<std::vector<int64_t>> &max_dim);
  std::vector<std::vector<int64_t>> GetInputDims();
  std::vector<std::vector<int64_t>> GetOutputDims();
  std::vector<int64_t> GetInputSizes();
  std::vector<int64_t> GetOutputSizes();
  std::vector<magicmind::DataType> GetInputDataTypes();
  std::vector<magicmind::DataType> GetOutputDataTypes();

  void Run(int n, Queue *queue);
  void Run(int n);
  void Destroy();

  std::vector<void *> InputPtrs() {
    std::vector<void *> ptrs;
    for (auto t : input_tensors_) {
      ptrs.push_back(t->GetMutableData());
    }
    return ptrs;
  }
  std::vector<void *> OutputPtrs() {
    std::vector<void *> ptrs;
    for (auto t : output_tensors_) {
      ptrs.push_back(t->GetMutableData());
    }
    return ptrs;
  }

  void H2D() { input_buffer_->H2D(); }
  void H2D(Queue *queue) { input_buffer_->H2D(queue); }
  void D2H() { output_buffer_->D2H(); }
  void D2H(Queue *queue) { output_buffer_->D2H(queue); }
  void PrintMemInfo(float &meminfo);
  std::vector<void *> GetHostOutputData();
  std::vector<void *> GetHostInputData();

 private:
  int dev_id_   = 0;
  Queue *queue_ = nullptr;
  std::string model_path_;
  magicmind::IModel *model_     = nullptr;
  magicmind::IEngine *engine_   = nullptr;
  magicmind::IContext *context_ = nullptr;
  std::vector<magicmind::IRTTensor *> input_tensors_;
  std::vector<magicmind::IRTTensor *> output_tensors_;
  Buffers *input_buffer_  = nullptr;
  Buffers *output_buffer_ = nullptr;
};

#endif  // MODEL_RUNNER_H_
