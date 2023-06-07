# MMDetection

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 MMDetection框架下的网络模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本项目使用的网络模型来自GitHub开源项目[MMDetection](https://github.com/open-mmlab/mmdetection) 分支版本为v2.28.2,本项目支持的网络如下：

| 模型 | 配置文件 | 预训练模型| 图像尺寸 | 
| --------- | ---------- | ---------- | ---------- | 
| Mask_R-CNN | [Config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth)|800x800 |
| Faster_R-CNN | [Config](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) | 800x800 |
| RetinaNet | [Config](https://github.com/open-mmlab/mmdetection/tree/main/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) | 800x800 |
| SSD | [Config](https://github.com/open-mmlab/mmdetection/tree/main/configs/ssd/ssd512_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth) | 512x512 |
| Cascade_R-CNN | [Config](https://github.com/open-mmlab/mmdetection/blob/main/configs/cascade_rcnn/cascade-rcnn_r101-caffe_fpn_1x_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth) | 800x800 |
| HRNet | [Config](https://github.com/open-mmlab/mmdetection/blob/main/configs/hrnet/cascade-mask-rcnn_hrnetv2p-w32_20e_coco.py) | [Model](https://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth) | 800x800 |

下面将展示如何将MMDetection框架下的网络模型转换为MagicMind的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/mmdetection
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
用户需先在env.sh里面选择使用MMDetection的具体某一个模型，即设置`MMDETECTION_MODEL_NAME`,也可参照env.sh现有格式添加新的模型。
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
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmdetection_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size> <img_num>
bash run.sh ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmdetection_model_force_float32_true 1 1000
```

精度结果:
**以下示例为Mask_R-CNN精度结果**
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.550
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.276
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.661
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/detection/mmdetection && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Pytorch maskrcnn 模型转换为 MagicMind maskrcnn 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本模型推理代码支持复用MMDetection框架，在MMDetection加入对magicmind backend的支持，代码见export_model/magicmind.patch。

本例通过调用MMDetection源码下tools/deployment/test.py来完成模型推理和精度计算
test.py参数说明:

- `config`: COCO数据集配置文件
- `magicmind_model`: MagicMind 模型路径。
- `eval`: maskrcnn评估指标 可选bbox segm
- `out`: 结果输出文件 .pkl
- `device_id`: MLU Device ID
- `batch_size`: batch_size

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 MagicMind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --iterations 1000
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

通过5.2精度benchmark测试的脚本跑出 mmdetection 各个模型 在 coco val2017 数据集上5000张测试图片的 mAP 如下：

| Model | Batch_Size | Shape | Percision | BBox mAP(0.50:0.95)(%) | BBox mAP(0.50)(%) |
| --------- | ---------- | ---------- | --------- | --------- |--------- |
| Mask_R-CNN | 1 | 800x800 | force_float32 | 35.5 | 55.0 | 
| Faster_R-CNN | 1 | 800x800 | force_float32 | 35.0 | 55.0 | 
| RetinaNet | 1 | 800x800 | force_float32 | 33.5 | 51.5 | 
| SSD | 1 | 512x512 | force_float32 | 29.5 | 49.3 | 
| Cascade_R-CNN | 1 | 800x800 | force_float32 | 39.2 | 56.9 |
| HRNet | 1 | 800x800 | force_float32 | 39.8 | 56.2 |

*声明1 因MMDetection框架对于部分模型HW可变时不支持多batch的原因，MM模型在推理部分加入了多batch，需要固定输入图像尺寸(如800x800,用户也可自行修改为其他尺寸,原始模型输入HW可变，仅支持单batch),导致MLU精度同比GPU略有降低(**相同条件下可以和GPU精度对齐**)，若要复现官方精度，可在export_model/mmdetection/configs/下将模型配置文件修改为官方默认配置,且需要在推理代码中将batch_size改为1*

*声明2  因MMDetection框架和原始模型原因，SSD模型模型推理仅支持dynamic shape为false，Cascade模型推理仅支持batch_size为1*

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- coco 数据集下载链接： http://images.cocodataset.org/zips/val2017.zip
- openlab 开源目标检测框架 mmdetection: https://github.com/open-mmlab/mmdetection

