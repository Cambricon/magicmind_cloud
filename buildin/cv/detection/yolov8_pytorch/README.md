YOLOv8 PyTorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 YOLOv8 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1-模型概述)
- [前提条件](#2-前提条件)
- [快速使用](#3-快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4-高级说明)
  - [export_model 高级说明](#41-export_model-高级说明)
  - [gen_model 高级说明](#42-gen_model-高级说明)
  - [infer_python 高级说明](#43-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 YOLOv8 实现来自 github 开源项目 https://github.com/ultralytics/ultralytics 。下面将展示如何将该项目中 PyTorch 实现的 YOLOv8 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/yolov8_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `COCO_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd ${PROJ_ROOT_PATH}/gen_model
# bash run.sh <precision> <batch_size> <dynamic_shape> <magicmind_model>
# 指定您想输出的magicmind_model路径，例如./model
bash run.sh force_float32 16 true ${magicmind_model}
```

### 3.5 执行推理

1. infer_python 执行推理

```bash
cd ${PROJ_ROOT_PATH}/infer_python
#bash run.sh <magicmind_model> <image_num> <yolo_mode> <batch_size>
bash run.sh ${magicmind_model} 1000 val 16
```

使用 COCO API 计算精度，结果如下：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.653
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/detection/yolov8_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 export_model 高级说明

1.由于 MagicMind 最高支持 PyTorch 1.6.0 版本，此版本没有 SiLU 函数，所以要在环境中修改代码如下：

```bash
vim /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py
#进行如下修改
return F.silu(input, inplace=self.inplace) 替换为 return input * torch.sigmoid(input)
```

或者直接运行以下代码：

```bash
patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
```

2.当前代码修改官方仓库代码，以使程序能在寒武纪板卡上正常部署，如下所示：

```bash
cd ${PROJ_ROOT_PATH}/export_model/ultralytics
# 修改ultralytics，使用mm进行推理
git apply ${PROJ_ROOT_PATH}/export_model/yolov8_pytorch.patch
```

### 4.2 gen_model 高级说明

YOLOv8 PyTorch  模型转换为 YOLOv8 MagicMind  模型分成以下几步：

- 使用 官方导出模型接口将pt文件转成opset=11的onnx 模型，再解析为 MagicMind 网络结构。
- 模型量化。
- 添加模型配置。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.3 infer_python 高级说明

参数说明：

- `magicmind_model`: MagicMind 模型路径。
- `image_num`: 输入图像数量，如果设置为-1，则表示使用数据集内的全部图像。
- `yolo_mode`: yolo推理模式，val为验证，predict为预测。
- `batch_size`: 一次推理的图片张数。

**注意：**
当前predict模式会复制一份数据集，如果数据集有写权限，可以修改infer_python/run.sh，取消复制数据集并替换推理时指定的数据集。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${magicmind_model} --batch_size ${batch_size} --iterations 1000 --devices ${device_id} --input_dims 16,3,640,640
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

一键运行 benchmark 里的脚本，可得到 YOLOv8 在 COCO2017 数据集中的 mAP， 如下所示：

| Model  | Precision           | Batch_Size | mAP (0.5:0.95) | mAP (0.5) |
| ------ | ------------------- | ---------- | -------------- | --------- |
| YOLOv8 | force_float32       | 16          |     0.373      |   0.525   |
| YOLOv8 | force_float16       | 16          |     0.373      |   0.526   |
| YOLOv8 | qint8_mixed_float16 | 16          |     0.363      |   0.520   |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：[http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- COCO VAL2017 标签下载链接：[http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- YOLOv8 模型下载链接：https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
- YOLOv8 github 下载链接：https://github.com/ultralytics/ultralytics.git
