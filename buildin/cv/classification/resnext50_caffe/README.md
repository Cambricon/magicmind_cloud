# resnext50_caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 resnext50 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [下载数据集，模型](#33-下载数据集,模型)
  - [编译 MagicMind 模型](#34-编译-MagicMind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41gen_model高级说明)
  - [infer_python 高级说明](#42infer_python高级说明)
  - [infer_cpp 高级说明](#43infer_cpp高级说明)
- [精度和性能 benchmark](#5精度和性能benchmark)
  - [性能 benchmark 结果](#51-性能-benchmark-结果)
  - [精度 benchmark 结果](#52-精度-benchmark-结果)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 resnext50 模型来自https://github.com/soeaver/caffe-model/tree/08979265fc1e1931cef27f40038d492167165804/cls。
下面将展示如何将该项目中 Caffe 实现的 resnext50 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/resnext50_caffe
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `ILSVRC2012_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集,模型

模型和数据集需要手动下载，下载地址在当前README结尾。
下载之后模型需要放在环境变量 `MODEL_PATH` 指定目录下，数据集需要和 `ILSVRC2012_DATASETS_PATH` 一致，特别注意的是 `ILSVRC2012_DATASETS_PATH` 就是图片文件所在目录。
模型和数据集下载并放至指定目录后，执行如下命令：

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
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/resnext50_caffe/data/models/resnext50_caffe_model_force_float32_true_1
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <shape_mutable> <batch_size> <image_num>
bash run.sh force_float32 true 1 100
```

2.infer_cpp

```bash
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh <precision> <shape_mutable> <batch_size> <image_num>
bash run.sh force_float32 true 1 100
```

计算 top1 和 top5 精度:

```bash
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_labels.txt \
                                            --result_1_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result_1.txt \
                                            --result_5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result_5.txt \
                                            --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result.txt
 
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_labels.txt \
                                            --result_1_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result_1.txt \
                                            --result_5_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result_5.txt \
                                            --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result.txt
```

结果：

```bash
top1:  0.6
top5:  0.84
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1gen_model 高级说明

Caffe resnext50 模型转换为 MagicMind resnext50 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `caffe_model`: caffe 的权重路径。
- `prototxt`: caffe 的网络结构路径。
- `output_model`: 保存 MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行分类任务。
- `label_file`: 标签文件路径。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `input_width`: W。
- `input_height`: H。
- `device_id`: 设备号。

### 4.2infer_python 高级说明

概述：
本例使用 MagicMind PYTHON API 编写了名为 infer_python 的目标检测程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 resnet50 图像分类(图像预处理=>推理=>后处理)。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行分类任务。
- `image_num`: 输入图像的数量。
- `name_file`: imagenet 名称文件路径。
- `label_file`: 标签文件路径。
- `result_file`: 输入图像。
- `result_label_file`: 输出 label 文件。
- `result_top1_file`: top1 文件
- `result_top5_file`: top5 文件

### 4.3infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 resnext50 图像分类(图像预处理=>推理=>后处理)。

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
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 结果

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
./benchmark/perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本跑出 resnext50 在 IMAGENET2012 数据集上的 TOP1 和 TOP5 如下：

```bash
cd $PROJ_ROOT_PATH
./benchmark/eval.sh
```

| Model     | Precision         | Batch_Size | TOP1     | TOP5      |
| --------- | ------------------ | ---------- | -------- | --------- |
| resnext50 | force float32      | 1          | 0.64664  | 0.86638   |
| resnext50 | force float16      | 1          | 0.6463   | 0.8665    |
| resnext50 | int8 mixed float16 | 1          | 0.64118  | 0.8614    |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- resnext50 caffemodel 和 prototxt file 下载链接：https://github.com/soeaver/caffe-model/tree/08979265fc1e1931cef27f40038d492167165804/cls
- LSVRC_2012 验证集链接: https://image-net.org/challenges/LSVRC
