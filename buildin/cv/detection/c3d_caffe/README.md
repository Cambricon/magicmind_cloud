# c3d_caffe

MagicMind 是面向寒武纪 MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind 能将深度学习框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 c3d 网络的 Caffe 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本例使用的 c3d 实现来自 github 开源项目https://github.com/facebookarchive/C3D/tree/master/C3D-v1.1。 下面将展示如何将该项目中 Caffe 实现的 c3d 模型转换为 MagicMind 的模型。

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

```bash
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1
```

```bash
Generate model done, model save to c3d_caffe/data/models/c3d_caffe_model_force_float32_false_1
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

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 细节说明

Caffe c3d 模型转换为 MagicMind c3d 模型分成以下几步：

- 使用 MagicMind Parser 模块将 caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `CAFFEMODEL`: c3d caffe 的权重路径。
- `PROTOTXT`: c3d caffe 的网络结构路径。
- `MM_MODEL`: 保存 MagicMind 模型路径。
- `DATASET_DIR`: 校准数据文件路径。
- `QUANT_MODE`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `SHAPE_MUTABLE`: 是否生成可变 batch_size 的 MagicMind 模型。
- `BATCH_SIZE`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `DEV_ID`: 设备号。

### 4.2 infer_cpp 细节说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的视频检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 c3d 视频捕捉(视频帧采样=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- resized_w: 预处理相关参数。指定图像预处理中缩放大小。
- resized_h: 预处理相关参数。指定图像预处理中缩放大小。
- magicmind_model: MagicMind 模型路径。
- video_list: 输入视频列表文件，文件中每一行为一个视频文件路径。
- output_dir: 动作识别结果保存目录。每一个视频片段的 top5 识别结果将保存为一个 txt 文件。
- sampling_rate: 视频帧采样频率，默认为 2，意味着每两帧采样一帧。
- clip_step: 截取视频片段的移动步长，默认为-1，若为-1，clip_step 等于 CLIP_LEN \* sampling_rate。
- test_nums: 输入数据数量 默认-1 表示检查全部输入数据
- name_file: 真实标签对应名称文件
- result_file: 推理结果总文件
- result_label_file: 推理结果标签文件
- result_top1_file: top1 结果文件
- result_top5_file top5 结果文件

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

得到如下性能结果：

| Model | QuantMode_ShapeMutable_BatchSize | Throughput (qps) | MLU compute Latency Avg (ms) | 95% (ms) | 99% (ms) | MLU 板卡类型 |
| ----- | -------------------------------- | ---------------- | ---------------------------- | -------- | -------- | ------------ |
| C3D   | force_float32_false_1            | 26.4652          | 37.773                       | 54.082   | 60.456   | MLU370 S4    |
| C3D   | force_float32_false_4            | 71.2793          | 56.105                       | 78.788   | 82.389   | MLU370 S4    |
| C3D   | force_float32_false_8            | 100.814          | 79.34                        | 101.92   | 105.91   | MLU370 S4    |
| C3D   | force_float16_false_1            | 66.162           | 15.101                       | 33.546   | 37.718   | MLU370 S4    |
| C3D   | force_float16_false_4            | 192.492          | 20.767                       | 43.44    | 47.214   | MLU370 S4    |
| C3D   | force_float16_false_8            | 291.771          | 27.405                       | 50.877   | 55.092   | MLU370 S4    |
| C3D   | qint8_mixed_float16_false_1      | 252.96           | 3.9402                       | 6.127    | 28.382   | MLU370 S4    |
| C3D   | qint8_mixed_float16_false_4      | 514.704          | 7.7583                       | 25.211   | 34.051   | MLU370 S4    |
| C3D   | qint8_mixed_float16_false_8      | 671.644          | 11.898                       | 14.984   | 38.591   | MLU370 S4    |
| C3D   | force_float32_false_1            | 30.0278          | 33.29                        | 37.234   | 47.184   | MLU370 X4    |
| C3D   | force_float32_false_4            | 91.3226          | 43.789                       | 57.619   | 61.683   | MLU370 X4    |
| C3D   | force_float32_false_8            | 145.804          | 54.855                       | 70.214   | 74.407   | MLU370 X4    |
| C3D   | force_float16_false_1            | 80.3762          | 12.424                       | 19.16    | 30.125   | MLU370 X4    |
| C3D   | force_float16_false_4            | 261.985          | 15.253                       | 20.272   | 31.963   | MLU370 X4    |
| C3D   | force_float16_false_8            | 427.534          | 18.7                         | 28.095   | 38.22    | MLU370 X4    |
| C3D   | qint8_mixed_float16_false_1      | 322.206          | 2.9381                       | 3.383    | 9.355    | MLU370 X4    |
| C3D   | qint8_mixed_float16_false_4      | 752.547          | 5.2258                       | 7.167    | 18.346   | MLU370 X4    |
| C3D   | qint8_mixed_float16_false_8      | 1019.77          | 7.7997                       | 11.537   | 27.926   | MLU370 X4    |

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```

通过快速使用中 3.6 的脚本跑出 c3d 在 UCF101 testlist01 数据集上的 mAP 如下：
| Model | QuantMode_ShapeMutable_BatchSize | @Acc(Top 1) | @Acc(Top 5) |MLU 板卡类型 |
| --------- | ---------- | ---------- | --------- | ---------
| C3D | force_float32_false_1 | 0.805666| 0.995788 | MLU370 S4 |
| C3D | force_float32_false_4 | 0.809610| 0.995623 | MLU370 S4 |
| C3D | force_float32_false_8 | 0.805180| 0.995522 | MLU370 S4 |
| C3D | force_float16_false_1 | 0.805433| 0.995322 | MLU370 S4 |
| C3D | force_float16_false_4 | 0.805570| 0.995322 | MLU370 S4 |
| C3D | force_float16_false_8 | 0.806170| 0.995422 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_1 | 0.805371| 0.994343 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_4 | 0.802633| 0.994324 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_8 | 0.802212| 0.996747 | MLU370 S4 |
| C3D | force_float32_false_1 | 0.805398| 0.996723 | MLU370 X4 |
| C3D | force_float32_false_4 | 0.806398| 0.996323 | MLU370 X4 |
| C3D | force_float32_false_8 | 0.804398| 0.992156 | MLU370 X4 |
| C3D | force_float16_false_1 | 0.802267| 0.998622 | MLU370 X4 |
| C3D | force_float16_false_4 | 0.800311| 0.998423 | MLU370 X4 |
| C3D | force_float16_false_8 | 0.800268| 0.992313 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_1 | 0.803371| 0.994234 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_4 | 0.805626| 0.996522 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_8 | 0.806042| 0.993535 | MLU370 X4 |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- c3d caffemodel file 下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
- prototxt 下载链接: https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt
- UCF101 数据集下载链接： https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar
- UCF101 数据集标签下载连接：https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip

## 7.Release_Notes

@TODO
