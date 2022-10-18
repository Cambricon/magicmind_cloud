#include <mm_runtime.h>
#include <cnrt.h>
#include <fstream>
#include <string>
#include <vector>
#include "../include/utils.hpp"

std::string GetFileName(const std::string &abs_path) {
  auto slash_pos = abs_path.rfind('/');
  if (std::string::npos == slash_pos) {
    std::cout << "[" << abs_path << "] is not an absolute path." << std::endl;
    return "";
  }
  if (slash_pos == abs_path.size() - 1) {
    return "";
  }
  auto point_pos = abs_path.rfind('.');
  if (point_pos == slash_pos + 1) {
    std::cout << "[" << abs_path << "] is not a file path." << std::endl;
    return "";
  }
  return abs_path.substr(slash_pos + 1, point_pos - slash_pos - 1);
}

/**
 * @brief model's info
 * @param mm model
 */
void PrintModelInfo(magicmind::IModel *model)
{
  std::cout << "================== Model Info  ====================" <<std::endl;
  std::cout << "Input number : " << model->GetInputNum() << std::endl;
  for (int i = 0; i < model->GetInputNum(); ++i)
    std::cout << "input[" << i << "] : dimensions " << model->GetInputDimension(i)
              << ", data type [" << model->GetInputDataType(i) << "]" << std::endl;
  std::cout << "Output number : " << model->GetOutputNum() << std::endl;
  for (int i = 0; i < model->GetOutputNum(); ++i)
    std::cout << "output[" << i << "] : dimensions " << model->GetOutputDimension(i)
              << ", data type [" << model->GetOutputDataType(i) << "]" << std::endl;
}
