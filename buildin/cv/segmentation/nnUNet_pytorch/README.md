# nnUNet_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 nnUnet 网络的 pytorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [infer_cpp 高级说明](#42-infer_cpp-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 nnUNet 实现来自 github 开源项目[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)。
下面将展示如何将该项目中 Pytorch 实现的 nnUnet 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd $PROJ_ROOT_PATH/export_model
# bash run.sh <parameter_id>
bash run.sh 0
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
# bash run.sh <parameter_id> <precision> <shape_mutable> <batch_size>
bash run.sh 0 force_float32 true 4
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch/../../../../../magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch/data/models/magicmind_models/nnUNet_pytorch_model_force_float32_true_0
```

### 3.5 执行推理

1.infer_cpp

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <parameter_id> <precision> <shape_mutable>
bash run.sh 0 force_float32 true
```

结果：

```bash
mean acc: {'1': {'Accuracy': 0.9924555498839087, 'Dice': 0.05997619505711238, 'False Discovery Rate': 0.9398969012674252, 'False Negative Rate': 0.9401406950143562, 'False Omission Rate': 0.0038114257755176317, 'False Positive Rate': 0.003764768198557672, 'Jaccard': 0.032507702467033946, 'Negative Predictive Value': 0.9961885742244825, 'Precision': 0.06010309873257471, 'Recall': 0.05985930498564386, 'Total Positives Reference': 46750.75, 'Total Positives Test': 46248.55, 'True Negative Rate': 0.9962352318014425}}
see output_f/summary.json for detail.
```

2.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <parameter_id> <precision> <shape_mutable>
bash run.sh 0 force_float32 true
```

结果：

```bash
mean acc: {'1': {'Accuracy': 0.9998284953122978, 'Dice': 0.9795684234762259, 'False Discovery Rate': 0.013382446563681599, 'False Negative Rate': 0.026790641636659764, 'False Omission Rate': 0.00011868492221191553, 'False Positive Rate': 5.353563091441416e-05, 'Jaccard': 0.9619934742159175, 'Negative Predictive Value': 0.9998813150777881, 'Precision': 0.9866175534363183, 'Recall': 0.9732093583633402, 'Total Positives Reference': 46750.75, 'Total Positives Test': 46036.3, 'True Negative Rate': 0.9999464643690859}}
see /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch/../../../../../magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch/data/output/infer_python_output_force_float32_true_1bs_0/summary.json for detail.
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/cv/segmentation/nnUNet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

Pytorch nnUNet 模型转换为 MagicMind nnUNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 pt 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: nnUNet pt 的路径。
- `batch_size`: 生成可变模型时 batch_size 可以在设定的 dim range 内取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `output_model`: 保存 MagicMind 模型路径。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `calib_data_path`: nnUNet calib_data pt 的路径。
- `device`: 设备号, 默认 0。
- `device_type`: 设备类型：MLU370。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 nnUNet 图像分割(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `model_path`: nnUNet 原始模型所在目录。
- `data_folder`: input data 存放目录。
- `output_folder`: output data 存放目录。
- `ref_folder`: ref data 存放目录。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 nnUNet 图像分割(图像预处理=>推理=>后处理)。
参数说明:

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `model_path`: nnUNet 原始模型所在目录。
- `data_folder`: input data 存放目录。
- `output_folder`: output data 存放目录。
- `ref_folder`: ref data 存放目录。
- `softmax_output_dir`: 推理的 softmax_output 存放目录。

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

一键运行 benchmark 里的脚本跑出 nnUNet 在 Task02_Heart 数据集上的精度如下：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```

| Model  | Precision           | Batch_Size | Accuracy           | Dice               | Precision          | Recall             |
| ------ | ------------------- | ---------- | ------------------ | ------------------ | ------------------ | ------------------ |
| nnUNet | force_float32       | 4          | 0.9998284953122978 | 0.9795684234762259 | 0.9866175534363183 | 0.9732093583633402 |
| nnUNet | force_float16       | 4          | 0.9998284875707446 | 0.9795677021465865 | 0.9866147441626959 | 0.973210859157148  |
| nnUNet | qint8_mixed_float16 | 4          | 0.9997113755692585 | 0.9654162003826233 | 0.9591708264052261 | 0.9723522342030166 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- nnUNet GITHUB 下载链接：[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- 模型下载链接：[https://zenodo.org/record/4003545/files/Task002_Heart.zip?download=1](https://zenodo.org/record/4003545/files/Task002_Heart.zip?download=1)
- 数据集下载链接：[https://drive.google.com/uc?id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY](https://drive.google.com/uc?id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY)
