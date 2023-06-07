# maskrcnn_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 maskrcnn 网络的 Pytorch 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本例使用的 maskrcnn 实现来自 github 开源项目https://github.com/open-mmlab/mmdetection 项目使用的backbone配置文件为mask_rcnn_r50_fpn_1x_coco.py 下面将展示如何将该项目中 Pytorch 实现的 maskrcnn 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/maskrcnn_pytorch
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
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
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <shape_mutable>
bash run.sh force_float32 true
```

精度结果:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.588
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.660
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/detection/maskrcnn_pytorch && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch maskrcnn 模型转换为 MagicMind maskrcnn 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `onnx_model`: maskrcnn onnx 的网络模型路径。
- `mm_model`: 保存 MagicMind 模型路径。
- `input_size`: 输入图像尺寸大小 默认为 224x224
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer.py 的目标检测检测程序。infer.py 将展示如何使用 MagicMind Python API 构建高效的 maskrcnn 目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:

- infer.py: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- `config`: COCO数据集配置文件
- `model`: MagicMind 模型路径。
- `eval`: maskrcnn评估指标 可选bbox segm
- `out`: 结果输出文件 .pkl
- `device_id`: MLU Device ID

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```

通过5.2精度benchmark测试的脚本跑出 maskrcnn 在 coco val2017 数据集上5000张测试图片的 mAP 如下：
_待更新_
| Model | BATCH_SIZE | Percision | BBox mAP(0.50:0.95)(%) | BBox mAP(0.50)(%) |
| --------- | ---------- | ---------- | --------- | --------- |
| maskrcnn | 1 | fp32 | 38.1 | 58.8 | 

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- coco 数据集下载链接： http://images.cocodataset.org/zips/val2017.zip
- openlab 开源目标检测框架 mmdetection: https://github.com/open-mmlab/mmdetection
