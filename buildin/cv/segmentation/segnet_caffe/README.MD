# segnet_caffe

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

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集,模型

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/segmentation/segnet_caffe/../../../../../magicmind_cloud/buildin/cv/segmentation/segnet_caffe/data/models/segnet_caffe_model_force_float32_true
```

### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
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

- `caffe_mode`: SegNet 模型路径。
- `prototxt`: SegNet prototxt 路径。
- `output_model`: 保存 MagicMind 模型路径。
- `image_dir`: 数据集存放目录。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: batch_size, 默认 1。
- `device_id`: 设备号, 默认 0。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind CPP API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind CPP API 构建高效的 SegNet 图像分割(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `image_dir`: 数据集存放目录。
- `image_list`: 图片 file list。
- `output_dir`: 输出存放目录。
- `save_txt` : 是否保存输出 txt。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

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

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 SegNet 在 VOC2012 数据集上的 mIOU 如下：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
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
