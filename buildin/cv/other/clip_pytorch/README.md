# clip_onnx

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 clip 网络的 Onnx 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-MagicMind-模型)
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

本例使用的 clip 实现来自 github 开源项目[https://github.com/openai/CLIP](https://github.com/openai/CLIP)。 下面将展示如何将该项目中 onnx 实现的 clip 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/other/clip_pytorch
```

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

- 下载数据集

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
# 指定您想输出的magicmind_model路径，例如./model
bash run.sh ${magicmind_model} force_float32 1 true
```

结果：

```bash
Generate model done , model save to /data/models/clip_onnx_force_float32_true
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh  <magicmind_model>  <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000
```

结果:

```
top1 accuracy: 0.617288
top5 accuracy: 0.867889
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/other/clip_pytorch && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Onnx clip 模型转换为 MagicMind clip 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Pytorch 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `onnx_model`: clip Onnx 的权重路径。
- `mm_model`: 生成的模型文件
- `precision`: 精度模式，如 force_float32，force_float16
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer_python 的视频检测程序。infer_python 将展示如何使用 MagicMind Python API 构建高效的 clip 图像分类程序。其中程序主要由以下内容构成:

- infer.py: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- device_id: MLU 设备号
- batch_size: 模型 batch_size
- magicmind_model: MagicMind 模型路径。
- datasets_dir: 数据集路径
- acc_result: 精度结果输出文件
- test_nums: 输入数据数量 默认-1 表示检查全部输入数据

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --devices $DEV_ID --threads 1 --iterations 1000
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

通过快速使用中 3.7 的脚本跑出 clip 在 CIFAR100 数据集上的 topk1 和 topk5 如下：
| Model | Precision_ShapeMutable_BatchSize | topk1 | topk5 |
| --------- | ---------- | ---------- | --------- |
| clip | force_float32_true_32 | 0.6171| 0.8678 |
| clip | force_float16_true_32 | 0.6167| 0.8671 |
| clip | qint8_mixed_float16_true_32 | 0.5378| 0.8058 |


## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- clip gitHub: https://github.com/openai/CLIP
