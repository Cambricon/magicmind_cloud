# arcface_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,Caffe 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。这份仓库探讨如何将 Pytorch 人脸识别网络 arcface 转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1 模型概述)
- [前提条件](#2 前提条件)
- [快速使用](#3 快速使用)
  - [环境准备](#3.1 环境准备)
  - [下载仓库](#3.2 下载仓库)
  - [下载数据集，模型](#3.3 下载数据集,模型)
  - [目录结构](#3.4 目录结构说明)
  - [模型转换](#3.5 模型转换)
  - [生成 MagicMind 模型](#3.6 编译生成 MagicMind 模型)
  - [编译运行](#3.7 编译运行)
  - [一键运行](#3.8 一键运行)
- [高级说明](#4 高级说明)
  - [gen_model 代码解释](#4.1 gen_model 代码解释)
  - [infer_cpp 代码解释](#4.2 infer_cpp 代码解释)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6 免责声明)
- [Release notes](#7 Release_Notes)

## 1 模型概述

本例使用的 arcface 模型来自 github 开源项目 https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch。

下面展示如何将该项目中 Pytorch 框架下 arcface 模型转换为 MagicMind 的模型。

## 2 前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪计算版本 MLU370 S4 或 MLU370 X4，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3 快速使用

### 3.1 环境准备

```bash
下载MagicMind镜像："yellow.hub.cambricon.com/MagicMind/release/x86_64/MagicMind:0.13.0-x86_64-ubuntu18.04-py_3_7"
docker load -i xxx.tar.gz
docker run -it --name=dockername --network=host --cap-add=sys_ptrace -v /your/host/path/MagicMind:/MagicMind -v /usr/bin/cnmon:/usr/bin/cnmon --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl -w /MagicMind/ <image name> /bin/bash
```

### 3.2 下载仓库

```bash
# 下载仓库
https://gitee.com/cambricon/magicmind_cloud.git
cd magicmind_cloud/buildin/cv/classification/arcface_pytorch
```

运行前，请检查以下路径，或执行`source env.sh`：

```
NEUWARE_HOME=/usr/local/neuware \
PROJ_ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
MAGICMIND_CLOUD="$( cd $PWD/../../../../ && cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" \
DATASETS_PATH=/nfsdata/modelzoo/datasets/ijb \
MODEL_PATH=$PROJ_ROOT_PATH/data/models \
UTILS_PATH=$MAGICMIND_CLOUD/buildin/cv/utils \
MM_RUN_PATH=$NEUWARE_HOME/bin
```

### 3.3 下载数据集,模型

- 数据集

本例使用[IJB](https://www.nist.gov/itl/iad/ig/ijb-c-dataset-request-form)数据集对模型精度进行验证.

数据集参考下载地址:

https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view

或

https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg

请将 IJB 数据集下载并解压至`$DATASETS_PATH`目录.参考命令如下:

```
cd $DATASETS_PATH
tar -xf ijb-*.tar
```

- 下载模型权重
  本例使用 MS1MV3 训练的 backborn 为 r100 的 arcface 模型进行实验,模型权重下载链接可参考:

https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

或

https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC

请将 ms1mv3_arcface_r100_fp16 目录中的 backbone.pth 下载至`$PROJ_ROOT_PATH/data/models/`目录.

### 3.4 目录结构

```
.
|-- README.md
|-- benchmark
|-- data
|-- env.sh
|-- export_model
|-- gen_model
|-- infer_cpp
`-- run.sh
```

目录结构说明：

- benchmark: 提供 mm_run 测试脚本，用于测试该模型在不同输入规模、不同数据精度、不同硬件设备下的性能；同时提供精度验证功能；

- gen_model: 主要涉及模型量化和转为 mm engine 过程，要求内部能够一键执行完成该模块完整功能；

- data: 用于暂存测试结果，保存模型等;

- export_model: 用于生成 torch.jit.trace 模型;

- infer_cpp: 主要涉及该模型推理的端到端(包含前后处理)的 c++源代码、头文件、编译脚本和运行脚本等，要求内部能够一键编译和执行完成该模块完整功能；一般图像类及前后处理不复杂的网络，建议要有 c++的推理；

- run.sh: 顶层一键执行脚本，串联各个部分作为整个 sample 的一键运行脚本；

### 3.5 模型转换

```
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
```

### 3.6 编译生成 MagicMind 模型

```
cd $PROJ_ROOT_PATH/gen_model
#param: quant_mode batch_size
bash run.sh qint8_mixed_float16 128
```

注：

1. 本实例支持 batch_size 设置功能，已测试最大规模为 256。
2. 结果默认保存在$PROJ_ROOT_PATH/data/models 文件夹。
3. 有关更多生成 MagicMind 模型参数相关设置,请参考//TODO

### 3.7 编译运行

进入 infer_cpp 目录，在当前目录编译生成可执行文件`host_infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash build.sh
```

推理：

```bash
./bin/host_infer \
    --magicmind_model $PROJ_ROOT_PATH/data/models/arcface_qint8_mixed_float16_1.mm \
    --image_dir $DATASETS_PATH/IJBC/loose_crop \
    --image_list $DATASETS_PATH/IJBC/meta/ijbc_name_5pts_score.txt \
    --save_img true \
    --output_dir $PROJ_ROOT_PATH/data/images
#参数解析：
      --magicmind_model 输入模型路径
      --image_dir 测试图片路径
      --image_list 测试图片file_list
      --save_img 是否保存检测结果
      --output_dir 检测结果保存路径
```

### 3.8 一键运行

以上 3.2~3.7 的步骤，均可以通过`bash run.sh` 实现一键运行。

**注:**

- 1. 执行一键运行脚本,需用户确保已下载好模型和数据集;
- 2. 非云平台用户若需测试完整精度结果,需将本地完整的数据集路径软连接到$MAGICMIND_CLOUD/datasets 路径下;

## 4 高级说明

### 4.1 gen_model 代码解释

Pytorch arcface 模型转换为 MagicMind，其流程主要分为以下两步:

- 将原始 pth 模型通过 torch.jit.trace 生成固化模型(\*.pt).
- 使用 MagicMind Parser 模块将 torch.jit.trace 生成的 pt 文件解析为 MagicMind 网络结构.
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

gen_model.py 参数说明:

- `pt_model`: 转换后 pt 的路径。
- `image_dir`: 输入图像 file_list,保存输入图像的路径。
- `output_model_path`: 保存 MagicMind 模型路径。
- `quant_mode`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `batch_size`: batch 大小，默认为 1。

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
mm_run --MagicMind_model $PROJ_ROOT_PATH/data/models/arcface_qint8_mixed_float16_1.mm --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16_1.mm 1
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```

结果(全数据集)：

```bash
IJB-C(1E-5)          IJB—C(1E-4)
{'1e-5': '94.90208', '1e-4': '96.59968'}
```

## 6 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- IJB 数据集下载链接:https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
- IJB 数据集下载链接:https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg
- arcface 代码下载链接:https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- arcface backborn 模型下载链接:https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC
- InsightFace 代码路径:https://github.com/deepinsight/insightface

## 7 Release_Notes

@TODO
