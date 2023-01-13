# mobilenetv2_caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 mobilenetv2 网络的 Caffe 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
  - [infer_cpp 高级说明](#42-infer_cpp-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 mobilenetv2 实现来自 github 开源项目https://github.com/shicai/MobileNet-Caffe。 下面将展示如何将该项目中 Caffe 实现的 mobilenetv2 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/mobilenetv2_caffe
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `DATASETS_PATH`, 并且执行以下命令：

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
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
```

### 3.5 执行推理

1.infer_cpp

```bash
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
```

计算 top1 和 top5 精度:

```
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/force_float32_true_1
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $OUTPUT_DIR/eval_labels.txt \
                                            --result_1_file $OUTPUT_DIR/eval_result_1.txt \
                                            --result_5_file $OUTPUT_DIR/eval_result_5.txt \
                                            --top1andtop5_file $OUTPUT_DIR/eval_result.txt
```

结果:

```
top1 accuracy: 0.727455
top5 accuracy: 0.912826
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/classification/mobilenetv2_caffe && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Caffe mobilenetv2 模型转换为 MagicMind mobilenetv2 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `CAFFEMODEL`: mobilenetv2 caffe 的权重路径。
- `PROTOTXT`: mobilenetv2 caffe 的网络结构路径。
- `MM_MODEL`: 保存 MagicMind 模型路径。
- `DATASET_DIR`: 校准数据文件路径。
- `PRECISION`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `SHAPE_MUTABLE`: 是否生成可变 batch_size 的 MagicMind 模型。
- `BATCH_SIZE`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `DEV_ID`: 设备号。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的视频检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 mobilenetv2 目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:

- infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- device_id: MLU 设备号
- batch_size: 模型 batch_size
- magicmind_model: MagicMind 模型路径。
- image_dir: 数据集路径
- label_file:ground truth 文件
- output_dir:推理输出-画框图像路径
- result_file:推理结果输出文件 txt 格式
- result_label_file 推理结果输出 label 文件 txt 格式
- result_top1_file:top1 推理结果输出 label 文件 txt 格式
- result_top5_file:top5 推理结果输出 label 文件 txt 格式
- test_nums: 输入数据数量 默认-1 表示检查全部输入数据

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

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

通过快速使用中 3.6 的脚本跑出 mobilenetv2 在 imagenet2012 数据集上的选出的50000张图片 top1 和 top5 精度如下
| Model | BatchSize | Precision | top1(%) | top5(%) | 
| --------- | ---------- | ---------- | --------- | ---------|
| mobilenetv2 | 1 | fp32 | 70.15 | 90.09 | 
| mobilenetv2 | 1 | fp16 | 70.24 | 90.09 | 
| mobilenetv2 | 1 | int8_mixed_float16 | 70.44 | 90.00 | 

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- mobilenetv2 caffemodel file 下载链接：https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel
- mobilenetv2 prototxt file 下载链接：https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt
- ImageNet数据集下载链接: https://image-net.org/challenges/LSVRC/
