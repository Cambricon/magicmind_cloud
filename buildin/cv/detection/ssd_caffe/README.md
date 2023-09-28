# ssd_caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 SSD 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 SSD 实现来自 github 开源项目[https://github.com/chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)。下面将展示如何将该项目中 Caffe 实现的 SSD 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/ssd_caffe
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `VOC2007_DATASETS_PATH`, 并且执行以下命令：

```bash
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
# bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/detection/ssd_caffe/../../../../../magicmind_cloud/buildin/cv/detection/ssd_caffe/data/models/ssd_caffe_model_force_float32_true
```

### 3.5 执行推理

1.infer_cpp

```bash
cd $PROJ_ROOT_PATH/infer_cpp
# bash run.sh <precision> <shape_mutable> <save_img>
bash run.sh force_float32 true true
```

VOC2007 计算精度:

```bash
python $UTILS_PATH/compute_voc_mAP.py --path $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/voc_preds/ \
                                      --devkit_path $VOC2007_DATASETS_PATH/VOCdevkit 2>&1 |tee $PROJ_ROOT_PATH/data/output/$infer_cpp_output_force_float32_true_1/log_eval
```

结果：

```bash
VOC07 metric? No
AP for aeroplane = 0.8945
AP for bicycle = 0.8953
AP for bird = 0.7977
AP for boat = 0.7328
AP for bottle = 0.5479
AP for bus = 0.8945
AP for car = 0.6938
AP for cat = 0.9771
AP for chair = 0.7580
AP for cow = 0.8358
AP for diningtable = 0.9044
AP for dog = 0.9614
AP for horse = 0.9181
AP for motorbike = 0.8778
AP for person = 0.8038
AP for pottedplant = 0.6153
AP for sheep = 0.7494
AP for sofa = 0.9786
AP for train = 0.9256
AP for tvmonitor = 0.8516
Mean AP = 0.8307
```

2.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <precision> <shape_mutable> <save_img>
bash run.sh force_float32 true true
```

VOC2007 计算精度:

```bash
python $UTILS_PATH/compute_voc_mAP.py --path $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/voc_preds/ \
                                      --devkit_path $VOC2007_DATASETS_PATH/VOCdevkit 2>&1 |tee $PROJ_ROOT_PATH/data/output/$infer_python_output_force_float32_true_1/log_eval
```

结果:

```bash
VOC07 metric? No
AP for aeroplane = 0.8945
AP for bicycle = 0.8953
AP for bird = 0.7977
AP for boat = 0.7328
AP for bottle = 0.5479
AP for bus = 0.8945
AP for car = 0.6938
AP for cat = 0.9771
AP for chair = 0.7580
AP for cow = 0.8358
AP for diningtable = 0.9044
AP for dog = 0.9614
AP for horse = 0.9181
AP for motorbike = 0.8778
AP for person = 0.8038
AP for pottedplant = 0.6153
AP for sheep = 0.7494
AP for sofa = 0.9786
AP for train = 0.9256
AP for tvmonitor = 0.8516
Mean AP = 0.8307
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/cv/detection/ssd_caffe && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

Caffe SSD 模型转换为 MagicMind SSD 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `caffe_model`: SSD Caffe 的权重路径。
- `prototxt`: SSD Caffe 的网络结构路径。
- `output_model`: 保存 MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行分类任务。
- `label_file`: 标签文件路径。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以在设定的 dim range 内取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `device_id`: 设备号。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 SSD 目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.hpp, infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。
- post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像保存路径，程序对后缀为 jpg 的图片执行目标检测任务。
- `output_dir`: 检测结果文件保存路径。
- `save_img`: 是否将结果可视化保存成图片。

### 4.3 infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 SSD(图像预处理=>推理=>后处理)。

参数说明：

- `MAGICMIND_MODEL`: MagicMind 模型路径。
- `devkit_path`: 输入图像路径。
- `result_path`: 检测结果文件保存路径。
- `show`: 是否将结果可视化保存成图片。

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

一键运行 benchmark 里的脚本跑出 SSD 在 VOC2007 数据集上的 mAP 如下：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```

| Model | Precision           | Batch_Size | mAP    |
| ----- | ------------------- | ---------- | ------ |
| SSD   | force_float32       | 1          | 0.7239 |
| SSD   | force_float16       | 1          | 0.7236 |
| SSD   | qint8_mixed_float16 | 1          | 0.7199 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- SSD caffemodel file 下载链接：[https://github.com/chuanqi305/MobileNet-SSD/raw/97406996b1eee2d40eb0a00ae567cf41e23369f9/mobilenet_iter_73000.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/97406996b1eee2d40eb0a00ae567cf41e23369f9/mobilenet_iter_73000.caffemodel)
- SSD prototxt file 下载链接：[https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt)
- voc 数据集下载链接：[http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
- 计算 voc mAP 的脚本下载链接：[https://raw.githubusercontent.com/luliyucoordinate/eval_voc/361b1953891827b2342b6d6ce92b66a31855cb0e/eval_voc.py](https://raw.githubusercontent.com/luliyucoordinate/eval_voc/361b1953891827b2342b6d6ce92b66a31855cb0e/eval_voc.py)
