#include "utils.hpp"

#include <fstream>
#include <string>
#include <vector>

void PrintModelInfo(magicmind::IModel *model) {
  LOG(INFO) << "==================Model info===================";
  LOG(INFO) << "Input number : " << model->GetInputNum();
  for (int i = 0; i < model->GetInputNum(); ++i)
    LOG(INFO) << "input[" << i << "] : dimensions " << model->GetInputDimension(i)
        << ", data type [" << model->GetInputDataType(i) << "]";
  LOG(INFO) << "Output number : " << model->GetOutputNum();
  for (int i = 0; i < model->GetOutputNum(); ++i)
    LOG(INFO) << "output[" << i << "] : dimensions " << model->GetOutputDimension(i)
        << ", data type [" << model->GetOutputDataType(i) << "]";
}

// The implementation of this function shows the limitations of the current program on the model.
bool CheckModel(magicmind::IModel *model) {
  if (model->GetInputNum() != 1) {
    LOG(ERROR) << "Input number is [" << model->GetInputNum() << "].";
    return false;
  }
  if (model->GetOutputNum() != 1) {
    LOG(ERROR) << "Output number is [" << model->GetOutputNum() << "].";
    return false;
  }
  if (model->GetInputDimension(0).GetDimsNum() != 5) {
    LOG(ERROR) << "Input dimensions number [" << model->GetInputDimension(0).GetDimsNum() << "].";
    return false;
  }
  if (model->GetInputDimension(0)[4] != 3) {
    LOG(ERROR) << "Input channel number [" << model->GetInputDimension(0)[4] << "].";
    return false;
  }
  if (model->GetInputDataType(0) != magicmind::DataType::FLOAT32) {
    LOG(ERROR) << "Input data type is [" << model->GetInputDataType(0) << "].";
    return false;
  }
  if (model->GetOutputDataType(0) != magicmind::DataType::FLOAT32) {
    LOG(ERROR) << "Output data type is [" << model->GetOutputDataType(0) << "].";
    return false;
  }
  return true;
}

std::string GetFileName(const std::string &abs_path) {
    auto slash_pos = abs_path.rfind('/');
    LOG_IF(FATAL, std::string::npos == slash_pos)
        << "[" << abs_path << "] is not an absolute path.";
    if (slash_pos == abs_path.size() - 1) {
        // directory
        return "";
    }
    auto point_pos = abs_path.rfind('.');
    LOG_IF(FATAL, point_pos == slash_pos + 1)
        << "[" << abs_path << "] is not a file path.";
    return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

std::vector<std::string> LoadLabels(const std::string &path) {
    std::ifstream ifs(path);
    LOG_IF(FATAL, !ifs.is_open()) << "Open label file failed. path : "
        << path;
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(ifs, line)) labels.emplace_back(std::move(line));
    ifs.close();
    return labels;
}


