# ArcFace_PyTorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,Caffe 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。这份仓库探讨如何将 Pytorch 人脸识别网络 arcface 转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [下载数据集和模型](#33-下载数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 代码解释](#41-gen_model-代码解释)
  - [infer_cpp 代码解释](#42-infer_cpp-代码解释)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 结果](#51-性能-benchmark-结果)
  - [精度 benchmark 结果](#52-精度-benchmark-结果)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 arcface 模型来自 github 开源项目 https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch。

下面展示如何将该项目中 Pytorch 框架下 arcface 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/arcface_pytorch
```


在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `IJB_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集和模型

本例使用[IJB](https://www.nist.gov/itl/iad/ig/ijb-c-dataset-request-form)数据集对模型精度进行验证，使用 MS1MV3 训练的 backbone 为 r100 的 arcface 模型进行实验。
模型和数据集需要手动下载，下载地址在当前README结尾。
注意下载模型时，请下载 ms1mv3_arcface_r100_fp16 目录中的 backbone.pth
下载之后模型需要放在环境变量 `MODEL_PATH` 指定目录下，数据集需要和 `IJB_DATASETS_PATH` 一致。
模型和数据集下载并放至指定目录后，执行如下命令：

```
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd ${PROJ_ROOT_PATH}/gen_model
# bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
# 指定您想输出的magicmind_model路径，例如./model
bash run.sh ${magicmind_model} force_float32 1 true
```

### 3.5 执行推理 

```bash
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000
```

### 3.6 一键运行

以上 3.2~3.5 的步骤，均可以通过运行 `cd magicmind_cloud/buildin/cv/classification/arcface_pytorch && bash run.sh`实现一键执行。

## 4.高级说明

### 4.1 gen_model 代码解释

Pytorch arcface 模型转换为 MagicMind，其流程主要分为以下两步:

- 将原始 pth 模型通过 torch.jit.trace 生成固化模型(\*.pt).
- 使用 MagicMind Parser 模块将 torch.jit.trace 生成的 pt 文件解析为 MagicMind 网络结构.
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

gen_model.py 参数说明:

- `image_dir`: 输入图像 file_list,保存输入图像的路径。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)
### 4.2 infer_cpp 代码解释

概述:
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 arcface 人脸识别程序(图像预处理=>推理=>后处理)。相关代码存放在 infer_cpp 目录下可供参考。其中程序主要由以下内容构成:

- infer.hpp, infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/eval.sh
```

结果(IJBC 全数据集)：

| Model    | Precision          | Batch_Size | 1E-5     | 1E-4     | 
| -------- | ------------------- | ---------- | -------- | -------- | 
| ArcFace | force_float32       | 64          | 95.07593  | 96.70195  | 
| ArcFace | force_float16       | 64          | 95.08105  | 96.70706  | 
| ArcFace | qint8_mixed_float16 | 64          | 94.94299  | 96.62014  | 

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- IJB 数据集下载链接:https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
- IJB 数据集下载链接:https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg
- arcface 代码下载链接:https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- arcface backbone 模型下载链接(请下载 ms1mv3_arcface_r100_fp16 目录中的 backbone.pth):https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC
- InsightFace 代码路径:https://github.com/deepinsight/insightface
