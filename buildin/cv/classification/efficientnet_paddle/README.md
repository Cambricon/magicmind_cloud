# efficientnet_paddle

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何将 efficientnetb7 网络的 paddle 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译MagicMind模型)
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

本例使用的 paddle 实现来自 github 开源项目[https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 。 下面将展示如何将该项目中 paddle 实现的 efficientnetb7 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/efficientnet_paddle
```

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

- 下载数据集和模型

```bash
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
```


### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ${MODEL_PATH}/efficientnet_paddle_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${MODEL_PATH}/efficientnet_paddle_model_force_float32_true 1 1000
```

结果:


```bash
top1:  0.835
top5:  0.967
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/classification/efficientnet_paddle && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

ONNX efficientnetb7 模型转换为 MagicMind efficientnetb7 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 efficientnetb7 图像分类(图像预处理=>推理=>后处理)。

参数说明:

- `device_id`: 设备号。
- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行分类任务。
- `image_num`: 输入图像的数量。
- `name_file`: imagenet 名称文件路径。
- `label_file`: 标签文件路径。
- `result_file`: 输入图像。
- `result_label_file`: 输出 label 文件。
- `result_top1_file`: top1 文件
- `result_top5_file`: top5 文件
- `batch_size`: 生成可变模型时 batch_size 可以在 dimension range 内取值，生成不可变模型时 batch_size 的取值需要对应 onnx 的输入维度。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 efficientnetb7 在 IMAGENET2012 数据集上的 TOP1 和 TOP5 如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

| Model    | Precision          | Batch_Size | TOP1     | TOP5     |
| -------- | ------------------- | ---------- | -------- | -------- |
| EfficientNetB7 | force_float32       | 1          | 0.8433  | 0.9691 |
| EfficientNetB7 | force_float16       | 1          | 0.84326  | 0.9691  |
| EfficientNetB7 | qint8_mixed_float16 | 1          | 0.82852  | 0.96256  |

efficientnetb7 magicmind模型测试参数设置与PaddleClas保持一致如下：
```
# config file: PaddleClas/ppcls/configs/ImageNet/EfficientNet/EfficientNetB7.yaml
# line78 - line95
Eval:
    dataset: 
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/val_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 632
        - CropImage:
            size: 600
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- efficientnetb7 paddle 模型：https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/model_zoo/efficientnet.py
- efficientnetb7 paddle 模型下载链接：https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/EfficientNetB7_infer.tar
- LSVRC_2012 验证集链接: https://image-net.org/challenges/LSVRC
