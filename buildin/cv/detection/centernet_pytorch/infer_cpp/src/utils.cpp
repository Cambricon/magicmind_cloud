#include <mm_runtime.h>
#include <cnrt.h>
#include <fstream>
#include <string>
#include <vector>
#include "../include/utils.hpp"

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

