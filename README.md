# ModelZoo Cloud

## 介绍

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。

MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX,Caffe等) 训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本仓库展示如何将CV分类、检测、分割、NLP、语音等场景的前沿和经典模型，通过MagicMind转换和优化，进而运行在基于MagicMind的推理加速引擎的寒武纪加速板卡上的示例程序，为开发者提供丰富的AI应用移植参考。

## 网络支持列表和链接
CV：
| MODELS  | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [Resnet50](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/cv/classification/resnet50_onnx) | ONNX | YES | YES | NO | YES |
| [VGG16](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/cv/classification/vgg16_caffe) | Caffe | YES | YES | YES | YES |
| [YOLOV5](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/cv/detection/yolov5_v6_1_pytorch) | PyTorch | YES | YES | YES | YES |
| [SSD](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/cv/detection/ssd_caffe) | Caffe | YES | YES | YES | YES |
| [Unet](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/cv/segmentation/nnUNet_pytorch) | PyTorch | YES | YES | NO | YES |

NLP:
| MODELS  | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [BERT](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/tree/v1.0/buildin/nlp/bert_qa_pytorch) | PyTorch | YES | YES | NO | YES |


## issues/wiki/forum跳转链接

## contrib指引和链接

## LICENSE
ModelZoo Cloud的License具体内容请参见[LICENSE](https://e.gitee.com/cambricon/repos/cambricon/magicmind-cloud/blob/v1.0/LICENSE)文件。

## 免责声明
ModelZoo仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于ModelZoo, ModelZoo也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在ModelZoo上，或者您希望更新ModelZoo中属于您的数据集或模型，请您通过Github或者Gitee中提交issue，您也可以联系ecosystem@cambricon.com告知我们。
