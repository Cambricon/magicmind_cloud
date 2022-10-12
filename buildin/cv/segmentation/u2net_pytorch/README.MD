# u2net_pytorch

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本sample探讨如何使用将U2Net网络的pytorch实现转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

## 目录
* [模型概述](#1.模型概述)
* [前提条件](#2.前提条件)
* [快速使用](#3.快速使用)
  * [环境准备](#3.1环境准备)
  * [下载仓库](#3.2下载仓库)
  * [下载数据集，模型](#3.3下载数据集,模型)
  * [编译MagicMind模型](#3.4编译MagicMind模型)
  * [执行推理](#3.5执行推理)
  * [一键运行](#3.6一键运行)
* [高级说明](#4.高级说明)
  * [gen_model细节说明](#4.1gen_model细节说明)
  * [infer_python细节说明](#4.2infer_python细节说明)
* [精度和性能benchmark](#5.精度和性能benchmark)
  * [性能benchmark结果](#5.1性能benchmark结果)
  * [精度benchmark结果](#5.2精度benchmark结果)
* [免责声明](#6.免责声明)
* [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的U2Net实现来自github开源项目https://github.com/xuebinqin/U-2-Net。
下面将展示如何将该项目中Pytorch实现的U2Net模型转换为MagicMind的模型。

## 2.前提条件

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本 MLU370 S4或 MLU370 X4，并安装好驱动(>=v4.20.6)；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

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

### 3.2下载仓库

```bash
# 下载仓库
git clone https://gitee.com/cambricon/magicmind_cloud.git
```
在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：
```bash
source env.sh
```

### 3.3下载数据集,模型
```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```
数据集MSRA-B下载地址：https://mmcheng.net/msra10k/
官方预训练模型下载地址：https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing

### 3.4编译MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 1
```

### 3.5执行推理

```bash 
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32 1
```

结果：

```bash 
average mae: 8.7251, max fmeasure: 4.2186
```

### 3.6一键运行
以上3.3~3.5的步骤也可以通过运行./run.sh来实现一键执行

## 4.高级说明

### 4.1gen_model细节说明
Pytorch u2net模型转换为MagicMind u2net模型分成以下几步：
* 使用MagicMind Parser模块将onnx文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `pt_model`: u2net pt的路径。
* `batch_size`: batch_size的取值需要对应pt的输入维度。
* `input_width`: W。
* `input_height`: H。
* `output_model`: 保存MagicMind模型路径。
* `quant_mode`: 量化模式，如force_float32，force_float16，qint16_mixed_float32。
* `file_list`: 用于量化的输入图片的路径。
* `device`: 设备号, 默认0。

### 4.2infer_python细节说明
概述：
本例使用MagicMind PYTHON API编写了名为infer_python的目标检测程序。infer_python将展示如何使用MagicMind PYTHON API构建高效的u2net图像分类(图像预处理=>推理=>后处理)。

参数说明:
* `magicmind_model`: MagicMind离线模型存放目录。
* `output_folder`: output data存放目录。
* `device_id`: MLU设备id, 默认0。
* `img_dir`: 数据集图片目录。
* `batch_size`: batch size。
* `save_img`: 是否报错结果为图片。
  
## 5.精度和性能benchmark

### 5.1性能benchmark结果
本仓库通过寒武纪提供的Magicmind性能测试工具mm_run展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH
./benchmark/perf.sh
```

得到如下性能结果：
| Model  | Quant_Mode | Batch_Size | Throughput (qps) | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| u2net | force_float32 | 1 | 34.5137 | MLU370 S4 |
| u2net | force_float32 | 4 | 54.8677 | MLU370 S4 |
| u2net | force_float32 | 8 | 58.3338 | MLU370 S4 |
| u2net | force_float16 | 1 | 64.6213 | MLU370 S4 |
| u2net | force_float16 | 4 | 121.846 | MLU370 S4 |
| u2net | force_float16 | 8 | 141.88  | MLU370 S4 |
| u2net | qint16_mixed_float32 | 1 | 53.3801 | MLU370 S4 |
| u2net | qint16_mixed_float32 | 4 | 86.3547 | MLU370 S4 |
| u2net | qint16_mixed_float32 | 8 | 96.3341 | MLU370 S4 |

| Model  | Quant_Mode | Batch_Size | Throughput (qps) | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| u2net | force_float32 | 1 | 37.7032 | MLU370 X4 |
| u2net | force_float32 | 4 | 68.5801 | MLU370 X4 |
| u2net | force_float32 | 8 | 77.643  | MLU370 X4 |
| u2net | force_float16 | 1 | 69.3775 | MLU370 X4 |
| u2net | force_float16 | 4 | 141.702 | MLU370 X4 |
| u2net | force_float16 | 8 | 174.42  | MLU370 X4 |
| u2net | qint16_mixed_float32 | 1 | 56.8866 | MLU370 X4 |
| u2net | qint16_mixed_float32 | 4 | 97.7327 | MLU370 X4 |
| u2net | qint16_mixed_float32 | 8 | 111.602 | MLU370 X4 |

### 5.2精度benchmark结果
一键运行benchmark里的脚本跑出u2net在MSRA-B数据集上的mae和fmeasure如下：

```bash
cd $PROJ_ROOT_PATH
./benchmark/eval.sh
```
| Model  | Quant_Mode | Batch_Size | Average MAE | Max Fmeasure | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| u2net | force_float32 | 1 | 8.7251 | 4.2186 | MLU370 S4 |
| u2net | force_float16 | 1 | 8.7322 | 4.2187 | MLU370 S4 |
| u2net | qint16_mixed_float32 | 1 | 8.7088 | 4.2185 | MLU370 S4 |

| Model  | Quant_Mode | Batch_Size | Average MAE | Max Fmeasure |MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| u2net | force_float32 | 1 | 8.7251 | 4.2186 | MLU370 X4 |
| u2net | force_float16 | 1 | 8.7322 | 4.2187 | MLU370 X4 |
| u2net | qint8_mixed_float16 | 1 | 8.7088 | 4.2185 | MLU370 X4 |

## 6. 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* U2Net GITHUB下载链接：https://github.com/xuebinqin/U-2-Net.git
* 模型下载链接：https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing
* 数据集下载链接：https://mmcheng.net/msra10k/

## 7.Release_Notes
@TODO
