# ModelZoo Cloud

## 介绍

MagicMind 是面向寒武纪 MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。

MagicMind 能将深度学习框架(Tensorflow,PyTorch,ONNX,Caffe 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本仓库展示如何将 CV 分类、检测、分割、NLP、语音等场景的前沿和经典模型，通过 MagicMind 转换和优化，进而运行在基于 MagicMind 的推理加速引擎的寒武纪加速板卡上的示例程序，为开发者提供丰富的 AI 应用移植参考。

## 网络支持列表和链接

CV：
| MODELS | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [Alexnet](buildin/cv/classification/alexnet_bn_caffe) | Caffe | YES | YES | YES | NO |
| [MobilenetV2](buildin/cv/classification/mobilenetv2_caffe) | Caffe | YES | YES | YES | NO |
| [MobilenetV3](buildin/cv/classification/mobilenetv3_pytorch) | Pytorch | YES | YES | YES | NO |
| [Resnet50](buildin/cv/classification/resnet50_onnx) | ONNX | YES | YES | NO | YES |
| [Resnext50](buildin/cv/classification/resnext50_caffe) | Caffe | YES | YES | YES | YES |
| [SENet50](buildin/cv/classification/senet50_caffe) | Caffe | YES | YES | YES | NO |
| [Squeezenet_v1_0](buildin/cv/classification/squeezenet_v1_0_caffe) | Caffe | YES | YES | YES | NO |
| [Squeezenet_v1_1_caffe](buildin/cv/classification/squeezenet_v1_1_caffe) | Caffe | YES | YES | YES | NO |
| [VGG16](buildin/cv/classification/vgg16_caffe) | Caffe | YES | YES | YES | YES |
| [C3D](buildin/cv/detection/c3d_caffe) | Caffe | YES | YES | YES | NO |
| [Centernet](buildin/cv/detection/centernet_pytorch) | PyTorch | YES | YES | YES | NO |
| [MaskRCNN](buildin/cv/detection/maskrcnn_pytorch) | Caffe | YES | YES | NO | YES |
| [SSD](buildin/cv/detection/ssd_caffe) | Caffe | YES | YES | YES | YES |
| [YOLOV3](buildin/cv/detection/yolov3_caffe) | Caffe | YES | YES | YES | NO |
| [YOLOV3 Tiny](buildin/cv/detection/yolov3_tiny_caffe) | Caffe | YES | YES | YES | NO |
| [YOLOV5](buildin/cv/detection/yolov5_v6_1_pytorch) | PyTorch | YES | YES | YES | YES |
| [Unet](buildin/cv/segmentation/nnUNet_pytorch) | PyTorch | YES | YES | NO | YES |
| [SEGnet](buildin/cv/segmentation/segnet_caffe) | Caffe | YES | YES | YES | NO |

NLP:
| MODELS | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [BERT](buildin/nlp/LanguageModeling/bert_qa_pytorch) | PyTorch | YES | YES | NO | YES |
| [Transformers](buildin/nlp/LanguageModeling/transformers_pytorch) | PyTorch | YES | YES | NO | YES |
| [TACOTRON2](buildin/nlp/SpeechSynthesis/tacotron2_onnx) | ONNX | YES | YES | NO | YES |

## issues/wiki/forum 跳转链接

## contrib 指引和链接

## LICENSE

ModelZoo Cloud 的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 免责声明

ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 ModelZoo, ModelZoo 也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 ModelZoo 上，或者您希望更新 ModelZoo 中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。
