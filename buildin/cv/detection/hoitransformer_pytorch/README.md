# hoitransformer_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 hoitransformer 网络的PyTorch 模型先转为 ONNX 模型，再转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [export_model 高级说明](#41-export_model-高级说明)
  - [gen_model 高级说明](#42-gen_model-高级说明)
  - [infer_python 高级说明](#43-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 hoitransformer 实现来自 github 开源项目[https://github.com/bbepoch/HoiTransformer](https://github.com/bbepoch/HoiTransformer.git) 中。下面将展示如何将该项目中 PyTorch 实现的 hoitransformer 模型先转为ONNX模型随后转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/hoitransformer_onnx
```

在开始运行代码前需要执行以下命令安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `HOIA_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd $PROJ_ROOT_PATH/export_model
# bash run.sh
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
# bash run.sh <precision> <batch_size>
bash run.sh force_float32 1
```

结果：

```bash
Generate model done, model save to /projs/model_zoo/magicmind_cloud/buildin/cv/detection/hoitransformer_onnx/../../../../../magicmind_cloud/buildin/cv/detection/hoitransformer_onnx/data/models/hoitransformer_force_float32_1.mm
```

### 3.5 执行推理

1. infer_python 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> 
bash run.sh force_float32
```
在final_report.txt中查看结果:
```bash
class 1 --- ap: 0.39944646716010324   max recall: 0.5997899159663865
class 2 --- ap: 0.8492089541673953   max recall: 0.8976396354288385
class 3 --- ap: 0.7136769651327071   max recall: 0.8190675017397355
class 4 --- ap: 0.5228963291914585   max recall: 0.6010928961748634
class 5 --- ap: 0.9423425489589803   max recall: 0.9579439252336449
class 6 --- ap: 0.9412196685667442   max recall: 0.9599777654252363
class 7 --- ap: 0.7608156178063856   max recall: 0.8196838347781744
class 8 --- ap: 0.9247333591498782   max recall: 0.9473684210526315
class 9 --- ap: 0.8297613020447868   max recall: 0.8933333333333333
class 10 --- ap: 0.4393326076747941   max recall: 0.4899497487437186
--------------------
mAP: 0.7323433819853233   max recall: 0.7985846977876563
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/detection/yolov5_v6_1_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 export_model 高级说明

1 使用下面的代码导出 onnx 模型文件。

```bash
python3 export_onnx.py --dataset_file=hoia --backbone=resnet50 --batch_size=1 --model_path=$MODEL_PATH/res50_hoia_a4caffe.pth
```

### 4.2 gen_model 高级说明

ONNX Hoitransformer 模型转换为 MagicMind Hoitranformer 模型分成以下几步：

- 使用 MagicMind Parser 模块将 ONNX 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `onnx_model`: ONNX 的路径。
- `datasets_dir`: 输入图像目录，程序对该目录下所有图片执行目标检测任务。
- `mm_model`: 保存 MagicMind 模型路径。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 的取值范围是实际batch size变化范围内的值，生成不可变模型时 batch_size 的取值需要对应 ONNX 的输入维度。

### 4.3 infer_python 高级说明

参数说明：

- `model_path`: MagicMind 模型路径。
- `dataset_file`: 指定测试数据集名称。
- `batch_size`: 指定测试batch_size。
- `log_dir`: 指定推理结果储存路径。

## 5. 精度和性能 benchmark
### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --input_dims $BATCH_SIZE,3,672,896--devices $DEV_ID --threads 1 --iterations 1000
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

一键运行 benchmark 里的脚本跑出 hoitransformer 在 hoia 数据集 中 MAP 如下：

| Model  | Precision           | Batch_Size | mAP(%)            | 
| ------ | ------------------- | ---------- | -------------- | 
| hoitransformer | force_float32       | 1         | 73.23          | 
| hoitransformer | force_float16       | 1         | 73.16          | 

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- hoia 数据集下载链接：[https://drive.google.com/uc?id=1TXxyK0bQI7y1r-zF_md43K78PjB74Kd7](https://drive.google.com/uc?id=1TXxyK0bQI7y1r-zF_md43K78PjB74Kd7)
- hoia 标签下载链接：[https://drive.google.com/uc?id=1OO7fE0N71pVxgUW7aOp7gdO5dDTmkr_v](https://drive.google.com/uc?id=1OO7fE0N71pVxgUW7aOp7gdO5dDTmkr_v)
- hoitransformer 模型下载链接：[https://drive.google.com/uc?id=1bNrFQ6a8aKBzwWc0MAdG2f24StMP9lhY](https://drive.google.com/uc?id=1bNrFQ6a8aKBzwWc0MAdG2f24StMP9lhY)
- hoitransformer github 下载链接：[https://github.com/bbepoch/HoiTransformer](https://github.com/bbepoch/HoiTransformer)
