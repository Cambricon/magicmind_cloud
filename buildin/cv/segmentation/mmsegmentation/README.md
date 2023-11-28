# MMSegmentation

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 MMSegmentation框架下的网络模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41-gen_model-高级说明)
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本项目使用的网络模型来自GitHub开源项目[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 分支版本为v0.30.0,本项目支持的网络如下：

| 模型 | 配置文件 | 预训练模型| 图像尺寸 | 
| --------- | ---------- | ---------- | ---------- | 
| OCRNet | [Config](https://github.com/open-mmlab/mmsegmentation/blob/0.x/configs/ocrnet/ocrnet_hr18s_512x1024_80k_cityscapes.py) | [Model](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth)| 1024x2048 |
| DeepLabV3 | [Config](https://github.com/open-mmlab/mmsegmentation/blob/0.x/configs/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes.py) | [Model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-57bb8425.pth) | 1024x2048 |
| UNet| [Config](https://github.com/open-mmlab/mmsegmentation/blob/0.x/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py) | [Model](https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth) | 512x1024 |


下面将展示如何将MMSegmentation框架下的网络模型转换为MagicMind的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/mmsegmentation
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
用户需先在env.sh里面选择使用MMSegmentation的具体某一个模型，即设置`MMDETECTION_MODEL_NAME`,也可参照env.sh现有格式添加新的模型。
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmsegmentation_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size>
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmsegmentation_model_force_float32_true 1
```

精度结果:
**以下示例为DeepLabV3精度结果**
```
+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|      road     | 97.85 | 98.85 |
|    sidewalk   | 82.71 | 90.04 |
|    building   | 90.32 |  95.6 |
|      wall     |  57.1 | 65.53 |
|     fence     | 57.36 | 67.91 |
|      pole     | 39.87 | 47.94 |
| traffic light | 56.53 | 69.88 |
|  traffic sign | 66.29 | 74.94 |
|   vegetation  | 89.52 | 95.86 |
|    terrain    | 61.01 | 72.82 |
|      sky      | 91.88 | 96.67 |
|     person    | 71.37 | 84.44 |
|     rider     | 55.77 | 69.19 |
|      car      | 92.52 | 96.77 |
|     truck     | 76.77 | 87.35 |
|      bus      | 84.07 | 91.69 |
|     train     | 72.05 | 75.29 |
|   motorcycle  | 58.17 | 69.25 |
|    bicycle    | 69.57 | 83.29 |
+---------------+-------+-------+
Summary:

+-------+-------+------+
|  aAcc |  mIoU | mAcc |
+-------+-------+------+
| 94.78 | 72.14 | 80.7 |
+-------+-------+------+
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/segmentation/mmsegmentation && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch maskrcnn 模型转换为 MagicMind maskrcnn 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本模型推理代码支持复用MMSegmentation框架，在MMSegmentation加入对magicmind backend的支持，代码见export_model/magicmind.patch。

本例通过调用MMSegmentation源码下tools/test.py来完成模型推理和精度计算
test.py参数说明:

- `config`: 模型配置文件
- `magicmind_model`: 推理模型路径。
- `eval`: maskrcnn评估指标 可选bbox segm
- `out`: 结果输出文件 .pkl
- `device_id`: MLU Device ID
- `batch_size`: batch_size
- `backend`: 后端选择 可选 pytorch | magicmind

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 MagicMind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --iterations 1000
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

通过5.2精度benchmark测试的脚本跑出 mmsegmentation 各个模型 在 cityscapes 数据集上500张测试图片的 mAP 如下：

| Model | Batch_Size | Shape | Percision | aAcc(%) | mIoU(%) | mAcc |
| --------- | ---------- | ---------- | --------- | --------- |--------- |--------- |
| OCRNet | 1 | 1024x2048 | force_float32 | 96.4 | 79.45 | 86.75 |
| OCRNet | 1 | 1024x2048 | force_float16 | 96.4 | 79.44 | 86.74 |
| OCRNet | 1 | 1024x2048 | qint8_mixed_float16 | 96.27 | 78.93 | 87.32 |
| DeepLabV3 | 1 | 1024x2048 | force_float32 | 96.07 | 78.36 | 85.56 |
| DeepLabV3 | 1 | 1024x2048 | force_float16 | 96.07 | 78.36 | 85.56 |
| DeepLabV3 | 1 | 1024x2048 | qint8_mixed_float16 | 95.99 | 77.73 | 84.9 |
| UNet| 1 | 1024x2048 | force_float32 | 94.91 | 69.1 | 76.76 |
| UNet| 1 | 1024x2048 | force_float16 | 94.91 | 69.1 | 76.76 |
| UNet| 1 | 1024x2048 | qint8_mixed_float16 | 94.81 | 68.82 | 76.14 |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- cityscapes 数据集下载链接：https://www.cityscapes-dataset.com/file-handling/?packageID=1
- openlab 开源语义分割框架 mmsegmentation: https://github.com/open-mmlab/mmsegmentation
