# fsanet_tensorflow

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 FSA-Net 网络的 Tensorflow 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本例使用的 fsanet 实现来自 github 开源项目[https://github.com/shamangary/FSA-Net](https://github.com/shamangary/FSA-Net)。 下面将展示如何将该项目中 Tensorflow 实现的 fsanet 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/other/fsanet_tensorflow
```

在开始运行代码前需要执行以下命令安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `AFLW2000_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

- 准备数据集

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

- 准备模型

原作者已经做了将原生 tensorflow checkpoint 转换为 pb格式的工作，具体过程可以参考[这里](https://github.com/shamangary/FSA-Net/tree/master/pre-trained/converted-models), 我们使用pb来转成MagicMind模型。

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 false 1
```

结果：

```bash
Generate model done, model save to magicmind_cloud/buildin/cv/other/fsanet_tensorflow/data/mm_model/fsanet_capsule_force_float32_false_1
Generate model done, model save to magicmind_cloud/buildin/cv/other/fsanet_tensorflow/data/mm_model/fsanet_nos_capsule_force_float32_false_1
Generate model done, model save to magicmind_cloud/buildin/cv/other/fsanet_tensorflow/data/mm_model/fsanet_var_capsule_force_float32_false_1
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <shape_mutable> <batch_size> <image_num>
bash run.sh force_float32 false 1 100
```

计算精度:

```
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <shape_mutable> <batch_size> <image_num>
bash run.sh force_float32 false 1 1969
```

结果:

```
 --------------------------------------------------------------------------------
 MAE = 5.07, [yaw,pitch,roll] = [4.50, 6.08, 4.64]
--------------------------------------------------------------------------------
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/cv/other/fsanet_tensorflow && bash run.sh` 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Tensorflow fsanet 模型转换为 MagicMind fsanet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Tensorflow pb 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `tf_pb`: tensorflow pb模型的路径。
- `mm_model`: 保存 MagicMind 模型路径。
- `datasets_dir`: 校准数据文件路径。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer_python 的视频检测程序。infer_python 将展示如何使用 MagicMind Python API 构建高效的 fsanet 头部姿态预测(图像加载=>推理=>图像后处理)。其中程序主要由以下内容构成:

- infer.py: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序加载该目录下处理好的npz格式的人脸图像。
- `image_num`: 输入图像数量。
- `device_id`: 设备号。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE  --iterations 1000
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

通过使用中 5.2 的脚本跑出 fsanet 在 AFLW2000 数据集上的 MAE 如下：

| Model  | Precision           | Batch_Size | MAE(°)  | yaw(°) | pitch(°) | roll(°) |
| ------ | ------------------- | ---------- | --------|--------|--------- | ------- | 
| FSA-Net| force_float32       | 1          |   5.07  |   4.50 |   6.08   |  4.64   |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- FSA-Net github: https://github.com/shamangary/FSA-Net
- 模型权重和测试数据均下载自FSA-Net github。
