# ocrnet

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 ocrnet 从 pytorch 转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [gen_model 高级说明](#41-gen_model-高级说明)
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 ocrnet 实现来自 github 开源项目[https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py)。

下面将展示如何将该项目中 pytorch 实现的 ocrnet 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/ocrnet_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `CITYSCAPES_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备模型代码和数据集

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```
**注意:** [cityscapes](https://www.cityscapes-dataset.com/downloads/)数据集需要自行登陆并注册账号进行下载。
### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision>
bash run.sh force_float32
```

### 3.5 执行推理

计算精度：
```bash
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/segmentation/ocrnet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

模型转换分成以下几步：
- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `onnx_mode`: ocrnet onnx 模型路径。
- `output_model`: 保存 MagicMind 模型路径。
- `image_dir`: cityscapes数据集目录。
- `device_id`: 量化时使用的设备id。
- `precision`: 精度模式，如 force_float32，force_float16, qint8_mixed_float16。
- `input_width`: 模型输入的宽。
- `input_height`: 模型输入的高。
- `batch_size`: 输入的 batch 数。
  注:onnx模型是通过mmsegmentation提供的工具将pyotrch模型转换而来. 具体见export_model目录下的脚本。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind python API 编写了一个测试精度的代码。infer.py 将展示如何使用 MagicMind python API 实现图像分割(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `config`: 用于测试精度的配置文件，这里延用了原始pytorch下的配置。
- `data_root`: cityscapes数据集目录。
- `json_file`: 保存计算结果的文件。
- `device_id`: 推理时使用的设备id。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 ocrnet 在 cityscapes 数据集上的 mIOU 如下：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```
| Model        | Precision           | Shape_Mutable(H/W) | mIoU   |
| ------------ | ------------------- | ------------------ | ------ |
| ocrnet | force_float32       | false               | 79.45 |
| ocrnet | force_float16       | false               | 79.44 |
| ocrnet | qint8_mixed_float16 | false               | 78.93 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- ocrnet 权重下载链接: [https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth](https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes/ocrnet_hr18_512x1024_160k_cityscapes_20200602_191001-b9172d0c.pth)
- cityscapes 数据集下载链接: [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)
