# MMAction2

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 MMAction2框架下的网络模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本项目使用的网络模型来自GitHub开源项目[MMAction2](https://github.com/open-mmlab/mmaction2) 分支版本为v0.24.1
,本项目支持的网络如下：

| 模型 | 配置文件 | 预训练模型| 图像尺寸 | 
| --------- | ---------- | ---------- | ---------- | 
| I3D |i3d_r50_32x2x1_100e_kinetics400_rgb.py | [Model](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth)| 30 3 32 256 256 |
| TSM | tsm_r50_1x1x8_50e_kinetics400_rgb.py | [Model](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | 8 3 224 224 |


下面将展示如何将MMAction2框架下的网络模型转换为MagicMind的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/other/mmaction2
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
用户需先在env.sh里面选择使用MMAction2的具体某一个模型，即设置`MMDETECTION_MODEL_NAME`,也可参照env.sh现有格式添加新的模型。
source env.sh
```
随后，需要安装依赖项文件：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
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
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmaction2_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size>
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmaction2_model_force_float32_true 1
```

精度结果:
**以下示例为I3D精度结果**
```
Evaluating top_k_accuracy ...

top1_acc	0.7264
top5_acc	0.9069
top1_acc: 0.7264
top5_acc: 0.9069
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/other/mmaction2 && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch mmaction2 模型转换为 MagicMind mmaction2 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本模型推理代码支持复用MMAction2框架，在MMAction2加入对magicmind backend的支持，代码见export_model/magicmind.patch。

本例通过调用MMAction2源码下tools/test.py来完成模型推理和精度计算
test.py参数说明:

- `config`: 模型配置文件
- `magicmind_model`: 推理模型路径。
- `eval`: mmaction2评估指标 可选top_k_accuracy
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

通过5.2精度benchmark测试的脚本跑出 mmaction2 各个模型 在 cityscapes 数据集上500张测试图片的 mAP 如下：

| Model | Batch_Size | Shape | Percision | Top1(%) | Top1(%) |
| --------- | ---------- | ---------- | --------- | --------- |--------- |
| I3D | 1 | 30 3 32 256 256 | force_float32 | 72.64 | 90.69 |
| I3D | 1 | 30 3 32 256 256 | force_float16 | 72.64 | 90.69 |
| I3D | 1 | 30 3 32 256 256 | qint8_mixed_float16 | 70.00 | 88.33 |
| TSM | 1 | 8 3 224 224 | force_float32 | 68.47 | 88.06 |
| TSM | 1 | 8 3 224 224 | force_float16 | 68.47 | 88.06 |
| TSM | 1 | 8 3 224 224 | qint8_mixed_float16 | 68.01 | 87.78 |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- openlab mmaction2开源框架 mmaction2: https://github.com/open-mmlab/mmaction2
