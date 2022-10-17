# openpose_caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 openpose 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上，支持 BODY_25 及 COCO 两个模型。

## 目录

- [模型概述](#1.模型概述)
- [前提条件](#2.前提条件)
- [快速使用](#3.快速使用)
  - [环境准备](#3.1环境准备)
  - [下载仓库](#3.2下载仓库)
  - [下载数据集，模型](#3.3下载数据集,模型)
  - [编译 MagicMind 模型](#3.4编译MagicMind模型)
  - [执行推理](#3.5执行推理)
  - [一键运行](#3.6一键运行)
- [高级说明](#4.高级说明)
  - [gen_model 高级说明](#4.1gen_model高级说明)
  - [infer_cpp 高级说明](#4.2infer_cpp高级说明)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 openpose 实现来自 github 开源项目https://github.com/CMU-Perceptual-Computing-Lab/openpose。下面将展示如何将该项目中Caffe实现的openpose模型转换为MagicMind的模型。

## 2.前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算版本 MLU370 S4 或 MLU370 X4，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3.快速使用

### 3.1 环境准备

若基于寒武纪云平台环境可跳过该环节。否则需运行以下步骤：

1.下载 MagicMind(version >= 0.13.0)镜像(下载链接待开放)，名字如下：

magicmind_version_os.tar.gz

2.加载：

```bash
docker load -i magicmind_version_os.tar.gz
```

3.运行：

```bash
docker run -it --name=dockername --network=host --cap-add=sys_ptrace -v /your/host/path/MagicMind:/MagicMind -v /usr/bin/cnmon:/usr/bin/cnmon --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl -w /MagicMind/ magicmind_version_image_name:tag_name /bin/bash
```

### 3.2 下载仓库

```bash
# 下载仓库
git clone https://gitee.com/cambricon/magicmind_cloud.git
```

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirement.txt
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集,模型

```bash
cd $PROJ_ROOT_PATH/export_model
./run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
./run.sh qint8_mixed_float16 1
```

### 3.5 执行推理

infer_cpp 执行推理
编译 infer_cpp 目录，在当前目录输出 x86 可执行文件`bin/infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
./build.sh
```

推理：

```bash
./run.sh qint8_mixed_float16 1
```

使用 COCO API 计算精度:

```bash
python $UTILS_PATH/compute_coco_keypoints.py --ann_file $DATASETS_PATH/annotations/person_keypoints_val2017.json \
                                           --res_file $PROJ_ROOT_PATH/data/images/body25_qint8_mixed_float16_1/BODY_25 \
                                           --res2_file $PROJ_ROOT_PATH/data/images/coco_qint8_mixed_float16_1/COCO \
                                           --output_file $PROJ_ROOT_PATH/data/images/qint8_mixed_float16_1_eval
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.539
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.766
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.789
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.707
......
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.497
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.743
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.768
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.687

```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1gen_model 高级说明

Caffe openpose 模型转换为 MagicMind yolov5 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `prototxt`: caffe 的网络结构路径。
- `caffe_model`: caffe 的权重路径。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `output_model`: 保存 MagicMind 模型路径。
- `quant_mode`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `calibrate_list`: 输入图片文件列表，使用列表中文件做为输入进行模型量化。
- `device_id`: 设备号。

### 4.2infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的模型推理程序。其中程序主要由以下内容构成:

- infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。
- post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_list`: 数据集文件列表文件。
- `save_img`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。
- `output_dir`: 根据检测结果进行渲染后的图像或 COCO API 风格检测结果文件保存路径。
- `network`: 指定推理模型是 BODY_25 还是 COCO

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```

通过快速使用中 3.5 的脚本跑出 openpose 在 COCO2017 数据集上的 AP 如下：
| Model | Quant_Mode | Batch_Size | AP IoU=0.50 |MLU 板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BODY_25 | force_float32 | 1 | 0.779 | MLU370 S4 |
| BODY_25 | force_float16 | 1 | 0.780 | MLU370 S4 |
| BODY_25 | qint8_mixed_float16 | 1 | 0.766 | MLU370 S4 |
| COCO | force_float32 | 1 | 0.752 | MLU370 S4 |
| COCO | force_float16 | 1 | 0.752 | MLU370 S4 |
| COCO | qint8_mixed_float16 | 1 | 0.742 | MLU370 S4 |

| Model   | Quant_Mode          | Batch_Size | AP IoU=0.50 | MLU 板卡类型 |
| ------- | ------------------- | ---------- | ----------- | ------------ |
| BODY_25 | force_float32       | 1          | 0.779       | MLU370 X4    |
| BODY_25 | force_float16       | 1          | 0.780       | MLU370 X4    |
| BODY_25 | qint8_mixed_float16 | 1          | 0.766       | MLU370 X4    |
| COCO    | force_float32       | 1          | 0.752       | MLU370 X4    |
| COCO    | force_float16       | 1          | 0.752       | MLU370 X4    |
| COCO    | qint8_mixed_float16 | 1          | 0.742       | MLU370 X4    |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
- COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- YOLOV5M 模型下载链接：https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
- BODY_25 prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
- BODY_25 caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
- COCO prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
- COCO caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
- openpose GITHUB 下载链接：https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
- rapidjson 代码下载链接：https://github.com/miloyip/rapidjson.git

## 7.Release_Notes

@TODO
