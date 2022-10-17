# deeplabv3_tf

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 deeplabv3 从 tensorflow 转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [gen_model 细节说明](#4.1gen_model细节说明)
  - [infer_cpp 细节说明](#4.2infer_cpp细节说明)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 deeplabv3 实现来自 github 开源项目https://github.com/tensorflow/models/tree/master/research/deeplab。

下面将展示如何将该项目中 Tensorflow 实现的 deeplabv3 模型转换为 MagicMind 的模型。

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
cd magicmind_cloud/buildin/cv/segmentation/deeplabv3_tf
```

1.开始运行代码前需要先执行以下命令安装必要的库：

```bash
pip install -r requirements.txt
```

2.在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

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
bash run.sh force_float32 true
```

### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 true 1000
```

结果：

```bash

```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 bash run.sh 来实现一键执行

## 4.高级说明

### 4.1gen_model 细节说明

TensorFlow deeplabv3 模型转换为 MagicMind deeplabv3 模型分成以下几步：

- 使用 MagicMind Parser 模块将 tf 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `tf_mode`: deeplabv3 模型路径。
- `output_model_path`: 保存 MagicMind 模型路径。
- `image_dir`: 数据集存放目录。
- `file_list`: 数据集文件列表。
- `quant_mode`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 h/w 的 MagicMind 模型。

### 4.2infer_cpp 细节说明

概述：
本例使用 MagicMind CPP API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind CPP API 构建高效的 deeplabv3 图像分割(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 离线模型存放目录。
- `image_dir`: 数据集存放目录。
- `output_dir`: 输出存放目录。
- `file_list`: 图片 file list。
- `image_num`: 推理图片数量。
- `shape_mutable`: 是否是 h/w 可变的 MagicMind 模型。
- `save_img` : 是否保存输出。

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
cd $PROJ_ROOT_PATH
./benchmark/perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本跑出 deeplabv3_tf 在 VOC2012 数据集上的 mIOU 如下：

```bash
cd $PROJ_ROOT_PATH
./benchmark/eval.sh
```

| Model        | Quant_Mode          | Shape_Mutable(H/W) | VOC_mIOU           | MLU 板卡类型 |
| ------------ | ------------------- | ------------------ | ------------------ | ------------ |
| deeplabv3_tf | force_float32       | true               | 0.7665120921136525 | MLU370 S4    |
| deeplabv3_tf | force_float16       | true               | 0.7663800619173691 | MLU370 S4    |
| deeplabv3_tf | qint8_mixed_float16 | true               | 0.7630175641817623 | MLU370 S4    |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- deeplabv3 权重下载链接: http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
- VOC2012 数据集下载链接: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## 7.Release_Notes

@TODO
