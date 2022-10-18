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

// The implementation of this function shows the limitations of the current program on the model.
bool CheckModel(magicmind::IModel *model) {
    if (model->GetInputNum() != 1) {
        std::cout << "Input number is [" << model->GetInputNum() << "]." << std::endl;
        return false;
    }
    if (model->GetOutputNum() != 1) {
        std::cout << "Output number is [" << model->GetOutputNum() << "]." << std::endl;
        return false;
    }
    if (model->GetInputDimension(0)[3] != 3) {
        std::cout << "Input channels [" << model->GetInputDimension(0)[3] << "]." << std::endl;
        return false;
    }
    if (model->GetInputDataType(0) != magicmind::DataType::UINT8) {
        std::cout << "Input data type is [" << model->GetInputDataType(0) << "]." << std::endl;
        return false;
    }
    if (model->GetOutputDataType(0) != magicmind::DataType::FLOAT32) {
        std::cout << "Output data type is [" << model->GetOutputDataType(0) << "]." << std::endl;
        return false;
    }
    return true;
}