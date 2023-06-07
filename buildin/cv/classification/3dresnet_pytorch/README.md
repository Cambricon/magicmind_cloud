# 3D-ResNet PyTorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(TensorFlow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 3D-ResNet 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本例使用的 3D-ResNets 实现来自 [github 开源项目](https://github.com/kenshohara/3D-ResNets-PyTorch)。
下面将展示如何将该项目中 PyTorch 实现的 3D-ResNets 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/3dresnet_pytorch
```


在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `KINETICS_DATASETS_PATH`, 并且执行以下命令：

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

### 3.5 执行推理

```bash
cd ${PROJ_ROOT_PATH}/infer_python
#bash run.sh <magicmind_model> <batch_size> 
bash run.sh ${magicmind_model} 1 
```

结果：

```bash
load ground truth
number of ground truth: 33966
load result
number of result: 33966
calculate top-1 accuracy
top-1 accuracy: 0.5184007536948714
load ground truth
number of ground truth: 33966
load result
number of result: 33966
calculate top-5 accuracy
top-5 accuracy: 0.7543720190779014
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/classification/3dresnet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

PyTorch 3D-ResNet 模型转换为 MagicMind 3D-ResNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 PyTorch 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `image_dir`: kinetics数据集图片路径

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer_python 的视频分类程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 3D ResNet 图像分类(图像预处理=>推理=>后处理)。

参数说明:

- `video_path`: 输入数据集目录。
- `annotation_path`: annotation文件路径.
- `magicmind_model`: 保存 MagicMind 模型路径。
- `use_mlu`: 使用mlu设备进行推理
- `result_path`: 保存推理结果json文件的路径

**注意：**
本demo支持多batch推理，也支持多个视频同时推理。

（1）多batch推理的说明
当我们在讨论多batch推理时，此时的batch为程序运行过程中的`batch`，例如待推理tensor的shape[0]，与各目录中`run.sh`内的`batch_size`并不相同。此处的`batch`含义为图片数，即同时对多张图片进行图像分类。又由于本模型是通过从输入视频里抽帧得到待检测的图片，因此，即使每次处理的输入视频只有1个，实际进行推理的图片仍然极有可能不止1张（例如：大约有59%的概率会有19张图片同时进行推理，即batch有59%的概率为19）。

（2）多视频推理的说明
根目录的`run.sh`内的`batch_size`的含义是待同时处理的视频数，默认为1，即1次只处理1个视频。由于本程序`dim_range`最大设置为了`64`（每次送入推理的图片数量不能超过64张），且如前文提到，每个视频每次抽帧时，大部分时候都会抽取19张图片，因此，根目录的`run.sh`内的`batch_size`通常不能超过`3`.

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${magicmind_model} --batch_size ${batch_size} --devices ${device_id} --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 3D-ResNet 在 Kinetics 数据集上的准确率如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

| Model | Precision    | Batch_Size | top1(%) | top5(%) |
| ----- | ------------- | ---------- | -------- | --------- |
| 3D-ResNet  | force_float32 | 1     | 51.84 | 75.44 |
| 3D-ResNet  | force_float16 | 1     | 51.83 | 75.43 |
| 3D-ResNet  | qint8_mixed_float16 | 1   | 50.96 | 74.57 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- 3D-ResNet github: https://github.com/kenshohara/3D-ResNets-PyTorch
- 3D-ResNet PyTorch 模型：https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M
- Kinetic 数据集: https://github.com/cvdfoundation/kinetics-dataset 


