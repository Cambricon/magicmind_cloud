# transformers_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 transformers 网络的 Pytorch 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1.模型概述)
- [前提条件](#2.前提条件)
- [快速使用](#3.快速使用)
  - [环境准备](#3.1环境准备)
  - [下载仓库](#3.2下载仓库)
  - [准备数据集和模型](#3.3准备数据集和模型)
  - [编译 MagicMind 模型](#3.4编译MagicMind模型)
  - [执行推理](#3.5执行推理)
  - [一键运行](#3.6一键运行)
- [高级说明](#4.高级说明)
  - [gen_model 代码解释](#4.1gen_model代码解释)
  - [infer_cpp 代码解释](#4.2infer_python代码解释)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 transformers 实现来自 github 开源项目https://github.com/huggingface/transformers/tree/v4.10.2 下面将展示如何将该项目中 Pytorch 实现的 transformers 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/nlp/LanguageModeling/transformers_pytorch
```
在开始运行代码前需要执行以下命令安装必要的库：

```baah
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
bash run.sh force_float32 true 1
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32 true 1
```

结果:

```
accuracy:0.8774509803921569
f1:0.9134948096885814
combined_score:0.8954728950403692
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/nlp/LanguageModeling/transformers_pytorch && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch transformers 模型转换为 MagicMind transformers 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Pytorch 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: transformers Pytorch 的权重路径。
- `mm_model`: 生成的模型文件
- `precision`: 精度模式，如 force_float32，force_float16
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer_cpp 的视频检测程序。infer_cpp 将展示如何使用 MagicMind Python API 构建高效的 transformer 文本分类程序。其中程序主要由以下内容构成:

- infer.py: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- device_id: MLU 设备号
- batch_size: 模型 batch_size
- magicmind_model: MagicMind 模型路径。
- datasets_dir: 数据集路径
- acc_result: 精度结果输出文件
- test_nums: 输入数据数量 默认-1 表示检查全部输入数据

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据。可变模型需要用户指定input_dims或batch_size。

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

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```

通过快速使用中 3.6 的脚本跑出 transformers 在 coco val2017 数据集上 1000 张测试图片的 mAP 如下：
| Model | BATCH_SIZE | Percision | accuracy | f1 |MLU 板卡类型 |
| --------- | ---------- | ---------- | --------- | --------- | --------- |
| transformers | 1 | fp32 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 16 | fp32 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 32 | fp32 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 1 | fp16 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 16 | fp16 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 32 | fp16 | 87.74 | 91.34 | MLU370 S4 |
| transformers | 1 | fp32 | 87.74 | 91.34 | MLU370 X4 |
| transformers | 16 | fp32 | 87.74 | 91.34 | MLU370 X4 |
| transformers | 32 | fp32 | 87.74 | 91.34 | MLU370 X4 |
| transformers | 1 | fp16 | 87.74 | 91.34 | MLU370 X4 |
| transformers | 16 | fp16 | 87.74 | 91.34 | MLU370 X4 |
| transformers | 32 | fp16 | 87.74 | 91.34 | MLU370 X4 |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- huggingface gitHub: https://github.com/huggingface/transformers/tree/v4.10.2

## 7.Release_Notes

@TODO
