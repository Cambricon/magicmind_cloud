# MMPose

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 MMPose框架下的网络模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本项目使用的网络模型来自GitHub开源项目[MMPose](https://github.com/open-mmlab/mmpose) 分支版本为v0.30.0,本项目支持的网络如下：

| 模型 | 配置文件 | 预训练模型| 图像尺寸 | 
| --------- | ---------- | ---------- | ---------- | 
| HRNet | hrnet_w32_coco_512x512.py | [Model](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth)|  HW可变 |
| HRNet | res50_coco_512x512.py | [Model](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth)|  HW可变 |
| HRNet | mobilenetv2_coco_512x512.py | [Model](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth)|  HW可变 |


下面将展示如何将MMPose框架下的网络模型转换为MagicMind的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/other/mmpose
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
用户需先在env.sh里面选择使用MMPose的具体某一个模型，即设置`MMPOSE_MODEL_NAME`,也可参照env.sh现有格式添加新的模型。
source env.sh
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
bash run.sh ${MODEL_PATH}/${MMPOSE_MODEL_NAME}_mmpose_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

1.infer_python
```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size> <img_num>
bash run.sh ${MODEL_PATH}/${MMPOSE_MODEL_NAME}_mmpose_model_force_float32_true 1 ${img_num} 
```

精度结果:
**以下示例为HRNet精度结果**
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.654
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.863
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.720
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.587
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.757
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.710
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.891
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.760
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] =  0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.821
AP: 0.6542431114216686
AP (L): 0.7572654540631519
AP (M): 0.5865731021935185
AP .5: 0.8627627958332257
AP .75: 0.7195844540204901
AR: 0.7099181360201512
AR (L): 0.8208472686733556
AR (M): 0.6293362469270691
AR .5: 0.8912153652392947
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/other/mmpose && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch mmpose 模型转换为 MagicMind 模型分成以下几步:

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本模型推理代码支持复用MMPose框架，在MMPose加入对magicmind backend的支持，代码见export_model/magicmind.patch。

本例通过调用MMPose源码下tools/test.py来完成模型推理和精度计算
test.py参数说明:

- `config`: 模型配置文件
- `magicmind_model`: 推理模型路径。
- `eval`: 模型评估指标 可选mAP
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

通过5.2精度benchmark测试的脚本跑出 mmpose 各个模型 在 COCO2017 数据集上5000张测试图片的 AP AR 如下：

| Model | Batch_Size | Shape | Percision | AP | AR |
| --------- | ---------- | ---------- | --------- | --------- |--------- |
| HRNet | 1 | HW可变 | force_float32 | 65.42 | 70.99 |
| HRNet | 1 | HW可变 | force_float16 | 65.37 | 70.95 |
| HRNet | 1 | HW可变 | qint8_mixed_float16 | 62.49 | 68.65 |
| ResNet50 | 1 | HW可变 | force_float32 | 46.78 | 55.30 |
| ResNet50 | 1 | HW可变 | force_float16 | 46.60 | 55.30 |
| ResNet50 | 1 | HW可变 | qint8_mixed_float16 | 45.61 | 54.21 |
| MobileNetv2 | 1 | HW可变 | force_float32 | 38.13 | 47.40 |
| MobileNetv2 | 1 | HW可变 | force_float16 | 37.98 | 47.29 |
| MobileNetv2 | 1 | HW可变 | qint8_mixed_float16 | 37.83 | 47.34 |

*声明 MMPose框架自身限制，仅支持bs=1,推理时dynamic_shape为true*

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- coco 数据集下载链接： http://images.cocodataset.org/zips/val2017.zip
- openlab 开源语义分割框架 mmpose: https://github.com/open-mmlab/mmpose
