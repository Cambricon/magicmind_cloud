#include "utils.hpp"

#include <fstream>
#include <string>
#include <vector>

void PrintModelInfo(magicmind::IModel *model) {
    cout << "==================Model info===================" << endl;
    cout << "Input number : " << model->GetInputNum() << endl;
    for (int i = 0; i < model->GetInputNum(); ++i)
        cout<< "input[" << i << "] : dimensions " << model->GetInputDimension(i)
            << ", data type [" << model->GetInputDataType(i) << "]" << endl;
    cout << "Output number : " << model->GetOutputNum() << endl;
    for (int i = 0; i < model->GetOutputNum(); ++i)
        cout << "output[" << i << "] : dimensions " << model->GetOutputDimension(i)
            << ", data type [" << model->GetOutputDataType(i) << "]" << endl;
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

