# UNet_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 UNet 网络的 pytorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [UNet\_pytorch](#UNet_pytorch)
  - [目录](#目录)
  - [1. 模型概述](#1-模型概述)
  - [2. 前提条件](#2-前提条件)
  - [3. 快速使用](#3-快速使用)
    - [3.1 环境准备](#31-环境准备)
    - [3.2 下载仓库](#32-下载仓库)
    - [3.3 准备数据集和模型](#33-准备数据集和模型)
    - [3.4 编译 MagicMind 模型](#34-编译-magicmind-模型)
    - [3.5 执行推理](#35-执行推理)
    - [3.6 一键运行](#36-一键运行)
  - [4. 高级说明](#4-高级说明)
    - [4.1 gen\_model 高级说明](#41-gen_model-高级说明)
    - [4.2 infer\_python 高级说明](#42-infer_python-高级说明)
  - [5. 精度和性能 benchmark](#5-精度和性能-benchmark)
    - [5.1 性能 benchmark 测试](#51-性能-benchmark-测试)
    - [5.2 精度 benchmark 测试](#52-精度-benchmark-测试)
  - [6. 免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 UNet 实现来自 github 开源项目[https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)。
下面将展示如何将该项目中 Pytorch 实现的 UNet 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/UNet_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `CARVANA_DATASETS_PATH`, 并且执行以下命令：

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
# bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ../data/models/unet_carvana_model_qint8_mixed_float16_true qint8_mixed_float16 1 true
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/segmentation/UNet_pytorch/data/models/unet_carvana_model_qint8_mixed_float16_true
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ../data/models/unet_carvana_model_qint8_mixed_float16_true 1 508
```

结果：

```bash
Dice coefficient: 0.9906
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/cv/segmentation/UNet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

Pytorch UNet 模型转换为 MagicMind UNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 pt 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: UNet pt 的路径。
- `batch_size`: 生成可变模型时 batch_size 可以在设定的 dim range 内取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `output_model`: 保存 MagicMind 模型路径。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `calib_data_path`: 量化图片的路径。
- `device`: 设备号, 默认 0。
- `device_type`: 设备类型：MLU370。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的推理程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 UNet 图像分割(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `data_folder`: input data 存放目录。
- `output_folder`: output data 存放目录。
- `ref_folder`: ref data 存放目录。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 UNet 在 Carvana data 数据集上的精度如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

| Model                          | Precision           | Batch_Size | Dice coefficient  |
| ------------------------------ | ------------------- | ---------- | ----------------- |
| unet_carvana_scale0.5_epoch2   | force_float32       | 1          | 0.9914            |
| unet_carvana_scale0.5_epoch2   | force_float16       | 1          | 0.9914            |
| unet_carvana_scale0.5_epoch2   | qint8_mixed_float16 | 1          | 0.9906            |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- UNet GITHUB 下载链接：[https://github.com/milesial/Pytorch-Unet](https://github.com/milesial/Pytorch-Unet)
- 模型下载链接：[https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0)
- 数据集下载链接：[Carvana data](https://www.kaggle.com/c/carvana-image-masking-challenge/data)
