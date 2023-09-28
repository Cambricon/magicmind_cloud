# u2net_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 U2Net 网络的 pytorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [gen_model 代码解释](#41gen_model-代码解释)
  - [infer_python 代码解释](#42infer_python-代码解释)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 U2Net 实现来自 github 开源项目 https://github.com/xuebinqin/U-2-Net
下面将展示如何将该项目中 Pytorch 实现的 U2Net 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/u2net_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `MSRA_B_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

模型和数据集需要手动下载，下载地址在当前README结尾。
下载之后模型需要放在环境变量 `MODEL_PATH` 指定目录下，数据集需要和 `MSRA_B_DATASETS_PATH` 一致。
模型和数据集下载并放至指定目录后，执行如下命令：

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision> <batch_size>
bash run.sh force_float32 1
```

### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <batch_size>
bash run.sh force_float32 1
```

结果：

```bash
average mae: 8.7251, max fmeasure: 4.2186
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1gen_model 代码解释

Pytorch u2net 模型转换为 MagicMind u2net 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: u2net pt 的路径。
- `batch_size`: batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `output_model`: 保存 MagicMind 模型路径。
- `precision`: 精度模式，如 force_float32，force_float16，qint16_mixed_float32。
- `file_list`: 用于量化的输入图片的路径。
- `device`: 设备号, 默认 0。

### 4.2infer_python 代码解释

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 u2net 图像分类(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `output_folder`: output data 存放目录。
- `device_id`: MLU 设备 id, 默认 0。
- `img_dir`: 数据集图片目录。
- `batch_size`: batch size。
- `save_img`: 是否报错结果为图片。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
./benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 u2net 在 MSRA-B 数据集上的 mae 和 fmeasure 如下：

```bash
cd $PROJ_ROOT_PATH
./benchmark/eval.sh
```

| Model | Precision           | Batch_Size | Average MAE | Max Fmeasure |
| ----- | -------------------- | ---------- | ----------- | ------------ |
| u2net | force_float32        | 1          | 8.7251      | 4.2186       |
| u2net | force_float16        | 1          | 8.7322      | 4.2187       |
| u2net | qint16_mixed_float32 | 1          | 8.7088      | 4.2185       |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- U2Net GITHUB 下载链接：https://github.com/xuebinqin/U-2-Net.git
- 模型下载链接：https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing
- 数据集下载链接：https://mmcheng.net/msra10k/
