# ModelZoo Cloud

## 1.介绍

MagicMind 是面向寒武纪 MLU 的推理加速引擎。

MagicMind 能将 AI 框架(TensorFlow,PyTorch,ONNX,Caffe 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本仓库展示如何将 CV 分类、检测、分割、NLP、语音等场景的前沿和经典模型，通过 MagicMind 转换和优化，进而运行在基于 MagicMind 的推理加速引擎的寒武纪加速板卡上的示例程序，为开发者提供丰富的 AI 应用移植参考。

## 2.前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪 300 系列及以上的智能加速卡，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往[寒武纪开发者社区](https://developer.cambricon.com/)申请试用;

## 3.环境准备

若基于寒武纪云平台环境可跳过该环节。否则需运行以下步骤：

1.请前往[寒武纪开发者社区](https://developer.cambricon.com/)下载 MagicMind(version >= 1.0.1)镜像，名字如下：

magicmind_version_os.tar.gz, 例如 magicmind_1.0.1-1_ubuntu18.04.tar.gz

2.加载：

```bash
docker load -i magicmind_version_os.tar.gz
```

3.运行：

```bash
docker run -it --shm_size 10G --name=dockername \
           --network=host --cap-add=sys_ptrace \
           -v /your/host/path/MagicMind:/MagicMind \
           -v /usr/bin/cnmon:/usr/bin/cnmon \
           --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl \
           -w /MagicMind/ magicmind_version_image_name:tag_name /bin/bash
```

## 4.网络支持列表和链接

### CV：

#### Classification:

| MODELS                                                             | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------------------------------------------------------------ | --------- | --------- | --------- | --- | ------ |
| [AlexNet](buildin/cv/classification/alexnet_bn_caffe)              | Caffe     | YES       | YES       | YES | NO     |
| [ArcFace](buildin/cv/classification/arcface_pytorch)               | PyTorch   | YES       | YES       | YES | NO     |
| [CRNN](buildin/cv/classification/crnn_pytorch)                     | PyTorch   | YES       | YES       | NO  | YES    |
| [DenseNet121](buildin/cv/classification/densenet121_caffe)         | Caffe     | YES       | YES       | YES | NO     |
| [EfficientNet](buildin/cv/classification/efficientnet_paddle)              | Paddle    | YES       | YES       | NO  | YES    |
| [GoogleNet_bn](buildin/cv/classification/googlenet_bn_caffe)       | Caffe     | YES       | YES       | YES | YES    |
| [MobileNetV2](buildin/cv/classification/mobilenetv2_caffe)         | Caffe     | YES       | YES       | YES | YES    |
| [MobileNetV3](buildin/cv/classification/mobilenetv3_pytorch)       | PyTorch   | YES       | YES       | YES | NO     |
| [ResNet50](buildin/cv/classification/resnet50_onnx)                | ONNX      | YES       | YES       | NO  | YES    |
| [ResNet50](buildin/cv/classification/resnet50_pytorch)             | PyTorch   | YES       | YES       | NO  | YES    |
| [ResNet50](buildin/cv/classification/resnet50_paddle)              | Paddle    | YES       | YES       | NO  | YES    |
| [ResNext50](buildin/cv/classification/resnext50_caffe)             | Caffe     | YES       | YES       | YES | YES    |
| [SENet50](buildin/cv/classification/senet50_caffe)                 | Caffe     | YES       | YES       | YES | NO     |
| [SqueezeNet_v1_1](buildin/cv/classification/squeezenet_v1_1_caffe) | Caffe     | YES       | YES       | YES | NO     |
| [SwinTransformer](buildin/cv/classification/swintransformer_pytorch)|PyTorch   | YES       | YES       | NO  | YES    |
| [VGG16](buildin/cv/classification/vgg16_caffe)                     | Caffe     | YES       | YES       | YES | YES    |
| [Xception](buildin/cv/classification/xception_paddle)              | Paddle    | YES       | YES       | NO  | YES    |
| [3D-ResNet](buildin/cv/classification/3dresnet_pytorch)            | PyTorch   | YES       | YES       | NO  | YES    |


#### Detection:

| MODELS                                                        | FRAMEWORK  | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| -----------------------------------------------------         | ---------- | --------- | --------- | --- | ------ |
| [C3D](buildin/cv/detection/c3d_caffe)                         | Caffe      | YES       | YES       | YES | NO     |
| [CenterNet](buildin/cv/detection/centernet_pytorch)           | PyTorch    | YES       | YES       | YES | NO     |
| [DBNet](buildin/cv/detection/dbnet_pytorch)                   | PyTorch    | YES       | YES       | NO  | YES    |
| [Faster_R-CNN](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [Cascade_R-CNN](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [HoiTransformer](buildin/cv/detection/hoitransformer_pyTorch) | PyTorch    | YES       | YES       | NO  | YES    |
| [HRNet](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [Mask_R-CNN(Offical API)](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [Mask_R-CNN](buildin/cv/detection/maskrcnn_pytorch)         | PyTorch    | YES       | YES       | NO  | YES    |
| [PSENet](buildin/cv/detection/psenet_tensorflow)      | TensorFlow | YES       | YES       | NO  | YES    |
| [PPYOLOE](buildin/cv/detection/paddledetection)      | Paddle | YES       | YES       | NO  | YES    |
| [Retinaface](buildin/cv/detection/retinaface_pytorch)         | PyTorch    | YES       | YES       | YES | NO     |
| [RetinaNet](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [SSD](buildin/cv/detection/ssd_caffe)                         | Caffe      | YES       | YES       | YES | YES    |
| [SSD](buildin/cv/detection/mmdetection)         | PyTorch    | YES       | YES       | NO  | YES    |
| [YOLOv3](buildin/cv/detection/yolov3_caffe)                   | Caffe      | YES       | YES       | YES | NO     |
| [YOLOv3](buildin/cv/detection/yolov3_tensorflow)              | TensorFlow | YES       | YES       | YES | NO     |
| [YOLOv3](buildin/cv/detection/yolov3_paddle)                  | Paddle     | YES       | YES       | NO  | YES     |
| [YOLOv3 Tiny](buildin/cv/detection/yolov3_tiny_caffe)         | Caffe      | YES       | YES       | YES | NO     |
| [YOLOv4](buildin/cv/detection/yolov4_caffe)           | Caffe      | YES       | YES       | YES | NO     |
| [YOLOv5](buildin/cv/detection/yolov5_v6_1_pytorch)            | PyTorch    | YES       | YES       | YES | YES    |
| [YOLOv5 without plugin](buildin/cv/detection/yolov5_v7_0_without_plugin_pytorch)            | PyTorch    | YES       | YES       | NO | YES    |
| [YOLOv7](buildin/cv/detection/yolov7_pytorch)                 | PyTorch    | YES       | YES       | YES | YES    |
| [YOLOv8](buildin/cv/detection/yolov8_pytorch)      | PyTorch | YES       | YES       | NO  | YES    |

#### Segmentation:

| MODELS                                                    | FRAMEWORK  | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| --------------------------------------------------------- | ---------- | --------- | --------- | --- | ------ |
| [Deeplabv3](buildin/cv/segmentation/mmsegmentation)        | PyTorch    | YES       | YES       | NO  | YES    |
| [Deeplabv3](buildin/cv/segmentation/deeplabv3_tensorflow) | TensorFlow | YES       | YES       | YES | NO     |
| [nnUnet](buildin/cv/segmentation/nnUNet_pytorch)          | PyTorch    | YES       | YES       | NO  | YES    |
| [OCRNet(Offical API)](buildin/cv/segmentation/mmsegmentation)        | PyTorch    | YES       | YES       | NO  | YES    |
| [OCRNet](buildin/cv/segmentation/ocrnet_pytorch/)        | PyTorch    | YES       | YES       | NO  | YES    |
| [SegNet](buildin/cv/segmentation/segnet_caffe)            | Caffe      | YES       | YES       | YES | NO     |
| [UNet](buildin/cv/segmentation/unet_pytorch)              | PyTorch    | YES       | YES       | NO  | YES    |
| [UNet(MMSegmentation)](buildin/cv/segmentation/mmsegmentation)        | PyTorch    | YES       | YES       | NO  | YES    |
| [U2Net](buildin/cv/segmentation/u2net_pytorch)            | PyTorch    | YES       | YES       | NO  | YES    |

#### Others:

| MODELS                                      | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------------------------------------- | --------- | --------- | --------- | --- | ------ |
| [Clip](buildin/cv/other/clip_pytorch)       | PyTorch   | YES       | YES       | NO  | YES    |
| [FSANet](buildin/cv/other/fsanet_tensorflow)| TensorFlow| YES       | YES       | NO  | YES    |
| [HRNet](buildin/cv/other/mmpose)| PyTorch| YES       | YES       | NO  | YES    |
| [I3D](buildin/cv/other/mmaction2)| PyTorch| YES       | YES       | NO  | YES    |
| [MobileNetv2](buildin/cv/other/mmpose)| PyTorch| YES       | YES       | NO  | YES    |
| [Openpose](buildin/cv/other/openpose_caffe) | Caffe     | YES       | YES       | YES | NO     |
| [PaddleOCRv2](buildin/cv/other/paddleocr)| Paddle| YES       | YES       | NO  | YES    |
| [PaddleOCRv3](buildin/cv/other/paddleocr)| Paddle| YES       | YES       | NO  | YES    |
| [ResNet50](buildin/cv/other/mmpose)| PyTorch| YES       | YES       | NO  | YES    |
| [TSM](buildin/cv/other/mmaction2)| PyTorch| YES       | YES       | NO  | YES    |

### NLP:

#### LanguageModeling:

| MODELS                                                            | FRAMEWORK  | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ----------------------------------------------------------------- | ---------- | --------- | --------- | --- | ------ |
| [BERT](buildin/nlp/LanguageModeling/bert_squad_pytorch)           | PyTorch    | YES       | YES       | NO  | YES    |
| [BERT](buildin/nlp/LanguageModeling/bert_tensorflow)              | TensorFlow | YES       | NO        | NO  | YES    |
| [ROBERTA](buildin/nlp/LanguageModeling/roberta_pytorch)           | PyTorch    | YES       | YES       | NO  | YES    |
| [Roformer-sim](buildin/nlp/LanguageModeling/roformer-sim_tensorflow) | TensorFlow | YES       | YES    | NO  | YES    |
| [Transformers](buildin/nlp/LanguageModeling/transformers_pytorch) | PyTorch    | YES       | YES       | NO  | YES    |

#### SpeechSynthesis:

| MODELS                                                  | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------------------------------------------------- | --------- | --------- | --------- | --- | ------ |
| [TACOTRON2](buildin/nlp/SpeechSynthesis/tacotron2_onnx) | ONNX      | YES       | YES       | NO  | YES    |

#### SpeechRecognition:

| MODELS                                                  | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------------------------------------------------- | --------- | --------- | --------- | --- | ------ |
| [WeNet](buildin/nlp/SpeechRecognition/wenet_pytorch)    | PyTorch   | YES       | YES       | NO  | YES    |

#### Recommendation:
| MODELS                                                  | FRAMEWORK | MLU370-X4 | MLU370-S4 | CPP | PYTHON |
| ------------------------------------------------------- | --------- | --------- | --------- | --- | ------ |
| [xDeepFM](buildin/nlp/Recommendation/xdeepfm_paddle)    | Paddle   | YES       | YES       | NO  | YES    |


## 5.issues/wiki/forum 跳转链接

## 6.contrib 指引和链接

## 7.LICENSE

ModelZoo Cloud 的 License 具体内容请参见[LICENSE](LICENSE)文件。

## 8.免责声明

ModelZoo 仅提供公共数据集以及预训练模型的下载链接，公共数据集及预训练模型并不属于 ModelZoo, ModelZoo 也不对其质量或维护承担责任。请您在使用公共数据集和预训练模型的过程中，确保符合其对应的使用许可。

如果您不希望您的数据集或模型公布在 ModelZoo 上，或者您希望更新 ModelZoo 中属于您的数据集或模型，请您通过 Gitee 中提交 issue，您也可以联系ecosystem@cambricon.com告知我们。

## 9.Release Note
### v1.7:
- MagicMind支持版本1.7.0
- 修复部分网络README错误
- 删除对TensorFlow框架支持，详情见MagicMind r1.7.0 Release Note

### v1.6:
- MagicMind支持版本1.6.0
- 整理代码格式，统一文件命名规则，修复部分网络README错误
- CV : 新增yolov8_pytorch、yolov5_v7_1_non_detection_output_pytorch网络支持
- NLP : 新增xdeepfm_paddle网络支持

### v1.5:
- MagicMind支持版本1.5.0
- 修复tacotron2_onnx perf测试中模型错误问题

### v1.4:
- MagicMind支持版本1.3.0
- 新增gen_model以及infer时的公共组件函数:python_common, cpp_common
- cv/classification: 
    重构 3dresnet_pytorch/arcface_pytorch/crnn_pytorch/densenet121_caffe/efficientnet_paddle/googlenet_bn_caffe/mobilenetv2_caffe/mobilenetv3_pytorch/resnet50_onnx/resnet50_paddle/resnet50_pytorch/senet50_caffe/squeezenet_v1_1_caffe/swintransformer_pytorch/vgg16_caffe/xception_paddle/
    删除 squeezenet_v1_0_caffe
- cv/detection: 
    重构 c3d_caffe/centernet_pytorch/dbnet_pytorch/retinaface_pytorch/yolov3_caffe/yolov3_tensorflow/yolov3_tiny_caffe/yolov4_caffe/yolov5_v6_1_pytorch/yolov7_pytorch
    新增 mmdetection/paddledetection
- cv/segmentation: 
    重构 unet_pytorch
    新增 mmsegmentation
- cv/other: 
    重构 clip_pytorch/openpose_caffe/
    新增 mmaction2/mmpose/paddleocr
- nlp/LanguageModeling: 
    重构 bert_squad_pytorch/bert_tensorflow/roberta_pytorch/roformer-sim_tensorflow/
- nlp/SpeechRecognition: 
    重构 wenet_pytorch
- nlp/SpeechSynthesis: 
    重构 tacotron2_onnx
### v1.3:

- MagicMind支持版本1.0.1

- CV : 新增CRNN/Densenet121/GoogleNet_bn/DBNet/ResNet50(Paddle)/SwinTransformer/3D-ResNet/YoloV3(TensorFlow&Paddle)/YoloV7/HoiTransformer/YoloV4/PSENet/Openpose/Clip/FSANet网络的支持

- NLP : 新增Roberta/Roformer-sim/WeNet网络的支持

### v1.2:

- MagicMind支持版本0.14.0

- CV : 新增AlexNet/SENet50/ArgFace/MaskRCNN/RetinaFace/SegNet/Deeplabv3/U2Net/Openpose网络的支持

- NLP : 新增BERT(TensorFlow)/Transformers网络的支持

### v1.1:

- CV : 新增ResNext50/Squeezenet_v1_1/Squeezenet_v1_0/Centernet/YoloV3/YoloV3 Tiny/C3D/MobileNetV2/MobileNetV3网络的支持

- NLP : 新增Tacotron2网络的支持

### v1.0:

- MagicMind支持版本0.13.0

- CV : 新增ResNet50/VGG16/YoloV5/SSD/Unet网络的支持

- NLP : 新增BERT(PyTorch)网络的支持
