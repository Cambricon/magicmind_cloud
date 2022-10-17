# mobilenetv2_caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 mobilenetv2 网络的 Caffe 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
- [细节说明](#4.细节说明)
  - [gen_model 代码解释](#4.1gen_model代码解释)
  - [infer_cpp 代码解释](#4.2infer_cpp代码解释)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 mobilenetv2 实现来自 github 开源项目https://github.com/shicai/MobileNet-Caffe。 下面将展示如何将该项目中 Caffe 实现的 mobilenetv2 模型转换为 MagicMind 的模型。

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

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集,模型

- 下载数据集

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1
```

结果：

```bash
Generate model done, model save to mobilenetv2_caffe/data/mm_model/force_float32_false_1
```

### 3.5 执行推理

1.infer_cpp

```bash
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 false 1
```

计算精度:

```
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/$force_float32_$false_1/eval_labels.txt \
                                            --result_1_file $PROJ_ROOT_PATH/data/output/$force_float32_$false_1/eval_result_1.txt \
                                            --result_5_file $PROJ_ROOT_PATH/data/output/$force_float32_$false_1/eval_result_5.txt \
                                            --top1andtop5_file $PROJ_ROOT_PATH/data/output/$force_float32_$false_1/eval_result.txt
```

结果:

```
top1 accuracy: 0.727455
top5 accuracy: 0.912826
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 细节说明

Caffe mobilenetv2 模型转换为 MagicMind mobilenetv2 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `CAFFEMODEL`: mobilenetv2 caffe 的权重路径。
- `PROTOTXT`: mobilenetv2 caffe 的网络结构路径。
- `MM_MODEL`: 保存 MagicMind 模型路径。
- `DATASET_DIR`: 校准数据文件路径。
- `QUANT_MODE`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `SHAPE_MUTABLE`: 是否生成可变 batch_size 的 MagicMind 模型。
- `BATCH_SIZE`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `DEV_ID`: 设备号。

### 4.2 infer_cpp 细节说明

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

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```

通过快速使用中 3.6 的脚本跑出 mobilenetv2 在 imagenet1000 数据集上的 top1 和 top5 精度如下
| Model | QuantMode_ShapeMutable_BatchSize | top1 | top5 |MLU 板卡类型 |
| --------- | ---------- | ---------- | --------- | ---------
| mobilenetv2 | force_float32_false_1 | 0.727455| 0.912826 | MLU370 S4 |
| mobilenetv2 | force_float32_false_4 | 0.727912| 0.912651 | MLU370 S4 |
| mobilenetv2 | force_float32_false_8 | 0.726815| 0.911290 | MLU370 S4 |
| mobilenetv2 | force_float16_false_1 | 0.728457| 0.912826 | MLU370 S4 |
| mobilenetv2 | force_float16_false_4 | 0.728916| 0.912651 | MLU370 S4 |
| mobilenetv2 | force_float16_false_8 | 0.727823| 0.911290 | MLU370 S4 |
| mobilenetv2 | qint8_mixed_float16_false_1 | 0.7204| 0.9078 | MLU370 S4 |
| mobilenetv2 | qint8_mixed_float16_false_4 | 0.7248| 0.9076 | MLU370 S4 |
| mobilenetv2 | qint8_mixed_float16_false_8 | 0.7227| 0.9112 | MLU370 S4 |
| mobilenetv2 | force_float32_false_1 | 0.7274| 0.9128 | MLU370 X4 |
| mobilenetv2 | force_float32_false_4 | 0.7279| 0.9126 | MLU370 X4 |
| mobilenetv2 | force_float32_false_8 | 0.7268| 0.9112 | MLU370 X4 |
| mobilenetv2 | force_float16_false_1 | 0.7284| 0.9128 | MLU370 X4 |
| mobilenetv2 | force_float16_false_4 | 0.7289| 0.9126 | MLU370 X4 |
| mobilenetv2 | force_float16_false_8 | 0.7278| 0.9112 | MLU370 X4 |
| mobilenetv2 | qint8_mixed_float16_false_1 | 0.7204| 0.9078 | MLU370 X4 |
| mobilenetv2 | qint8_mixed_float16_false_4 | 0.7248| 0.9076 | MLU370 X4 |
| mobilenetv2 | qint8_mixed_float16_false_8 | 0.7227| 0.9112 | MLU370 X4 |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- mobilenetv2 caffemodel file 下载链接：https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel
- mobilenetv2 prototxt file 下载链接：https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt

## 7.Release_Notes

@TODO
