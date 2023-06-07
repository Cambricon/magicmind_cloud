# SegNet_Caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 SegNet 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [infer_cpp 高级说明](#42-infer_cpp-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 SegNet 实现来自 github 开源项目[https://github.com/alexgkendall/SegNet-Tutorial/tree/2d0457ca20a7d22a81f07316bc04b2f26992730c](https://github.com/alexgkendall/SegNet-Tutorial/tree/2d0457ca20a7d22a81f07316bc04b2f26992730c)。
下面将展示如何将该项目中 Caffe 实现的 SegNet 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/segmentation/segnet_caffe
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `VOC2012_DATASETS_PATH`, 并且执行以下命令：


```bash
source env.sh
```

### 3.3 下载数据集,模型

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
cd ${PROJ_ROOT_PATH}/infer_cpp
#bash run.sh <magicmind_model>  <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1449
```

精度计算

```bash
python $UTILS_PATH/compute_voc_mIOU_segnet.py --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1
```

结果：

```bash
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████| 1449/1449 [00:04<00:00, 302.06it/s]
mIOU : 0.8842283867496735
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/cv/segmentation/segnet_caffe && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

Caffe SegNet 模型转换为 MagicMind SegNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如Caffe）模型解析、MagicMind 配置设定等，这些公共组件以及公共参数如`batch_size`, `device_id`的说明详见：[python公共组件的README.md](../../../python_common/README.md)

大部分参数为公共参数，网络特定参数如下：
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行分割任务。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 `infer.cpp` 的基于 SegNet 网络的图像分割程序(图像预处理=>推理=>后处理)，可高效地完成图像分类任务。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `image_dir`: 数据集存放目录。
- `image_list`: 图片 file list。
- `output_dir`: 输出存放目录。
- `save_txt` : 是否保存输出 txt。

**注意：**
在`infer_cpp`内使用了一些公共的组件，例如`model_runner`、日志系统、device抽象等，这些公共组件的说明详见：[cpp公共组件的README.md](../../../cpp_common/README.md) 


## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${magicmind_model} --batch_size ${batch_size} --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 SegNet 在 VOC2012 数据集上的 mIOU 如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

| Model  | Precision           | Batch_Size | VOC_mIOU           |
| ------ | ------------------- | ---------- | ------------------ |
| SegNet | force_float32       | 1          | 0.8842283867496735 |
| SegNet | force_float16       | 1          | 0.8842193798620179 |
| SegNet | qint8_mixed_float16 | 1          | 0.8839324729070673 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- VOC2012 数据集下载链接: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
- segnet_pascal.caffemodel 模型下载链接: [http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel](http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel)
- segnet_pascal.prototxt 模型下载链接: [https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/2d0457ca20a7d22a81f07316bc04b2f26992730c/Example_Models/segnet_pascal.prototxt](https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/2d0457ca20a7d22a81f07316bc04b2f26992730c/Example_Models/segnet_pascal.prototxt)
- SetNet GITHUB 下载链接: [https://github.com/alexgkendall/SegNet-Tutorial/tree/2d0457ca20a7d22a81f07316bc04b2f26992730c](https://github.com/alexgkendall/SegNet-Tutorial/tree/2d0457ca20a7d22a81f07316bc04b2f26992730c)

