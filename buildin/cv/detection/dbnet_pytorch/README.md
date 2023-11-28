# dbnet_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 DBNet 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41-gen_model-高级说明)
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 DBNet 模型来自[https://github.com/MhLiao/DB](https://github.com/MhLiao/DB)。
下面将展示如何将该项目中 PyTorch 实现的 DBNet 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/dbnet_pytorch
```

在开始运行代码前需要先安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `TOTAL_TEXT_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd ${PROJ_ROOT_PATH}/gen_model
# bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
# 指定您想输出的magicmind_model路径，例如./model
bash run.sh ${magicmind_model} force_float32 1 true
```

### 3.5 执行推理(包括精度验证)

1.infer_python

```bash
cd ${PROJ_ROOT_PATH}/infer_python
# bash run.sh <magicmind_model>
bash run.sh ${magicmind_model}
```

运行结果：

```bash
precision : 0.883901 (300)
recall : 0.774062 (300)
fmeasure : 0.825343 (1)
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/detection/dbnet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

PyTorch DBNet 模型转换为 MagicMind DBNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 pytorch 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 DBNet 图像检测(图像预处理=>推理=>后处理)。

参数说明:

- `polygon`: output polygons if true
- `box_thresh`: The threshold to replace it in the representers
- `magicmind_model`: MagicMind 模型路径。
- `result_file`: 保存精度结果文件路径。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model ${MM_MODEL} --batch_size ${BATCH_SIZE} --devices ${DEV_ID} --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 DBNet 在 total_text 数据集上的精度如下：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/eval.sh
```

| Model | Precision           | Batch_Size | precision | recall   | fmeasure |
| ----- | ------------------- | ---------- | --------- | -------- | -------- |
| DBNet | force_float32       | 1          | 0.883901  | 0.774062 | 0.825343 |
| DBNet | force_float16       | 1          | 0.883841  | 0.773610 | 0.825060 |
| DBNet | qint8_mixed_float16 | 1          | 0.881409  | 0.769092 | 0.821429 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- DBNet totaltext_resnet18 模型下载链接： [https://drive.google.com/drive/folders/12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7](https://drive.google.com/drive/folders/12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7)

- total_text 验证集链接: [https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG](https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG) 和 [https://drive.google.com/uc?id=1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2&export=download](https://drive.google.com/uc?id=1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2&export=download)

- DBNet github 下载链接：[https://github.com/MhLiao/DB](https://github.com/MhLiao/DB)
