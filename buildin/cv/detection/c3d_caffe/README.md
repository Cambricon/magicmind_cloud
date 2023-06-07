# c3d Caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(TensorFlow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 c3d 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1.模型概述)
- [前提条件](#2.前提条件)
- [快速使用](#3.快速使用)
  - [环境准备](#3.1环境准备)
  - [下载仓库](#3.2下载仓库)
  - [下载数据集，模型](#3.3准备数据集和模型)
  - [编译 MagicMind 模型](#3.4编译MagicMind模型)
  - [执行推理](#3.5执行推理)
  - [一键运行](#3.6一键运行)
- [高级说明](#4.高级说明)
  - [gen_model 代码解释](#4.1gen_model代码解释)
  - [infer_cpp 代码解释](#4.2infer_cpp代码解释)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 c3d 实现来自 github 开源项目https://github.com/facebookarchive/C3D/tree/master/C3D-v1.1 下面将展示如何将该项目中 Caffe 实现的 c3d 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/c3d_caffe
```

在开始运行代码前需要先检查 env.sh 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `UCF101_DATASETS_PATH`,并且执行以下命令：

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

1.infer_cpp

```bash
cd ${PROJ_ROOT_PATH}/infer_cpp
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 500
```

计算精度:

```
OUTPUT_DIR=${PROJ_ROOT_PATH}/data/output/force_float32_true_1
python ${UTILS_PATH}/compute_top1_and_top5.py --result_label_file ${OUTPUT_DIR}/eval_labels.txt \
                                            --result_1_file ${OUTPUT_DIR}/eval_result_1.txt \
                                            --result_5_file ${OUTPUT_DIR}/eval_result_5.txt \
                                            --top1andtop5_file ${OUTPUT_DIR}/eval_result.txt
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/detection/c3d_caffe && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

c3d Caffe 模型转换为  c3d MagicMind 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
1. 在`gen_model.py`内使用了一些公共的组件，例如arg解析、第三方框架（如Caffe）模型解析、MagicMind 配置设定等，这些公共组件以及公共参数如`batch_size`, `device_id`的说明详见：[python公共组件的README.md](../../../python_common/README.md)
2.  该网络仅能在`cluster_num`为`1`时运行.


大部分参数为公共参数，网络特定参数如下：
- `image_dir`: 待推理视频所在的目录。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的视频检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 c3d 视频捕捉(视频帧采样=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.cpp: 高效率地将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- resized_w: 预处理相关参数。指定图像预处理中缩放大小。
- resized_h: 预处理相关参数。指定图像预处理中缩放大小。
- magicmind_model: MagicMind 模型路径。
- video_list: 输入视频列表文件，文件中每一行为一个视频文件路径。
- output_dir: 动作识别结果保存目录。每一个视频片段的 top5 识别结果将保存为一个 txt 文件。
- sampling_rate: 视频帧采样频率，默认为 2，意味着每两帧采样一帧。
- clip_step: 截取视频片段的移动步长，默认为-1，若为-1，clip_step 等于 CLIP_LEN \* sampling_rate。
- image_num: 输入数据数量 默认 0 表示使用全部数据
- name_file: 真实标签对应名称文件
- result_file: 推理结果总文件
- result_label_file: 推理结果标签文件
- result_top1_file: top1 结果文件
- result_top5_file top5 结果文件

**注意：**
在`infer_cpp`内使用了一些公共的组件，例如 MagicMind上下文创建、资源释放、模型推理等，这些公共组件的说明详见：[cpp公共组件的README.md](../../../cpp_common/README.md)

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${magicmind_model} --batch_size ${batch_size} --devices ${device_id} --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

通过快速使用中 3.6 的脚本跑出 c3d 在 UCF101 testlist01 数据集上500张测试视频的 mAP 如下：
| Model | BATCH_SIZE| Percision | @Acc(Top 1) | @Acc(Top 5) |
| --------- | ---------- | ---------- | --------- | --------- |
| c3d | 1 | fp32 |  85.53 | 98.00 |
| c3d | 1 | fp16 | 85.53 | 98.00 |
| c3d | 1 | int8_mixed_float16 | 85.53 | 97.76 |


## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- c3d caffemodel file 下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
- prototxt 下载链接: https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt
- UCF101 数据集下载链接： https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar
- UCF101 数据集标签下载连接：https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip
