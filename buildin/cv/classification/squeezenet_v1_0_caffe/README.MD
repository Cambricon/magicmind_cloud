# squeezenet_v1_0_caffe

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX等) 训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本sample探讨如何将squeezenet_v1_0_caffe网络的Caffe实现转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

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
* [细节说明](#4.细节说明)
  * [build_model细节说明](#4.1build_model细节说明)
  * [infer_cpp细节说明](#4.2infer_cpp细节说明)
* [精度和性能benchmark](#5.精度和性能benchmark)
  * [性能benchmark结果](#5.1性能benchmark结果)
  * [精度benchmark结果](#5.2精度benchmark结果)
* [免责声明](#6.免责声明)
* [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的squeezenet_v1_0_caffe模型来自https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1。
下面将展示如何将该项目中Caffe实现的squeezenet_v1_0_caffe模型转换为MagicMind的模型。

## 2.前提条件

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本 MLU370 S4或 MLU370 X4，并安装好驱动(>=v4.20.6)；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3.快速使用

### 3.1环境准备@TODO
将magicmind_0.13.0-1_ubuntu18.04.tar.gz镜像载入本机。
加载镜像前，请确认当前docker环境就绪，且版本为v19.03及以上。参考以下命令加载镜像:
```bash
docker load -i magicmind_0.13.0-1_ubuntu18.04.tar.gz
```
运行镜像：
```bash
docker run -it --name=dockername --network=host --cap-add=sys_ptrace -v /your/host/path/magicmind:/magicmind -v /usr/bin/cnmon:/usr/bin/cnmon --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl -w /magicmind/ <image name> /bin/bash
```

### 3.2下载仓库

```bash
# 下载仓库
git clone http://gitlab.software.cambricon.com/neuware/software/ae/ecosystem/modelzoo/magicmind_cloud.git
```
在开始运行代码前需要先检查env.sh里的环境变量，并且执行以下命令：
```bash
source env.sh
```

### 3.3下载数据集,模型
```bash
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh
```

### 3.4编译MagicMind模型

```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 true 1
```

结果：

```bash 
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/classification/squeezenet_v1_0_caffe/data/models/squeezenet_v1_0_caffe_model_force_float32_true_1
```

### 3.5执行推理

1.infer_cpp
```bash 
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 true 1 1 1000
```
结果：

```bash 
top5 result:
0 [brain coral]
1 [coral reef]
2 [puffer, pufferfish, blowfish, globefish]
3 [eel]
4 [electric ray, crampfish, numbfish, torpedo]
```
计算top1和top5精度:

```bash
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_labels.txt \
                                            --result_1_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result_1.txt \
                                            --result_5_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result_5.txt \
                                            --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_force_float32_true_1/eval_result.txt
```

结果：
```bash
top1:  0.3843843843843844
top5:  0.6186186186186187
```

### 3.6一键运行
以上3.3~3.5的步骤也可以通过运行bash run.sh来实现一键执行

## 4.高级说明

### 4.1gen_model细节说明
Caffe squeezenet_v1_0模型转换为MagicMind squeezenet_v1_0模型分成以下几步：
* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `caffe_model`: squeezenet_v1_0 caffe的权重路径。
* `prototxt`: squeezenet_v1_0 caffe的网络结构路径。
* `output_model`: 保存MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行分类任务。
* `label_file`: 标签文件路径。
* `quant_mode`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `shape_mutable`: 是否生成可变batch_size的MagicMind模型。
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。
* `input_width`: W。
* `input_height`: H。
* `device_id`: 设备号。

### 4.2infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的目标检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的squeezenet_v1_0图像分类(图像预处理=>推理=>后处理)。

参数说明:
* `device_id`: 设备号。
* `magicmind_model`: MagicMind模型路径。
* `image_dir`: 输入图像目录，程序对该目录下所有后缀为jpg的图片执行分类任务。
* `image_num`: 输入图像的数量。
* `name_file`: imagenet 名称文件路径。
* `label_file`: 标签文件路径。
* `result_file`: 输入图像。
* `result_label_file`: 输出label文件。
* `result_top1_file`: top1文件
* `result_top5_file`: top5文件
* `batch_size`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。

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
| Squeezenet_v1_0 | force_float32 | 1 | 885.413 | MLU370 S4 |
| Squeezenet_v1_0 | force_float32 | 4 | 3409.08 | MLU370 S4 |
| Squeezenet_v1_0 | force_float32 | 8 | 3688.93 | MLU370 S4 |
| Squeezenet_v1_0 | force_float16 | 1 | 2133.71 | MLU370 S4 |
| Squeezenet_v1_0 | force_float16 | 4 | 8146.96 | MLU370 S4 |
| Squeezenet_v1_0 | force_float16 | 8 | 10161.7 | MLU370 S4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 1 | 3798.44 | MLU370 S4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 4 | 14983.8 | MLU370 S4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 8 | 18448.8 | MLU370 S4 |

| Model  | Quant_Mode | Batch_Size | Throughput (qps) | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Squeezenet_v1_0 | force_float32 | 1 | 889.528 | MLU370 X4 |
| Squeezenet_v1_0 | force_float32 | 4 | 3331.51 | MLU370 X4 |
| Squeezenet_v1_0 | force_float32 | 8 | 6528.16 | MLU370 X4 |
| Squeezenet_v1_0 | force_float16 | 1 | 2142.19 | MLU370 X4 |
| Squeezenet_v1_0 | force_float16 | 4 | 8098.5 | MLU370 X4 |
| Squeezenet_v1_0 | force_float16 | 8 | 15797.7 | MLU370 X4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 1 | 3869.47 | MLU370 X4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 4 | 14950.7 | MLU370 X4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 8 | 29452.9 | MLU370 X4 |

### 5.2精度benchmark结果
一键运行benchmark里的脚本跑出Squeezenet v1.1在IMAGENET2012数据集上的TOP1和TOP5如下：

```bash
cd $PROJ_ROOT_PATH
./benchmark/eval.sh
```
| Model  | Quant_Mode | Batch_Size | TOP1 | TOP5 |MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Squeezenet_v1_0 | force_float32 | 1 | 0.350350 | 0.583584 | MLU370 S4 |
| Squeezenet_v1_0 | force_float16 | 1 | 0.348348 | 0.583584 | MLU370 S4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 1 | 0.355355  | 0.578579 | MLU370 S4 |

| Model  | Quant_Mode | Batch_Size | TOP1 | TOP5 |MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Squeezenet_v1_0 | force_float32 | 1 | 0.350350 | 0.583584 | MLU370 X4 |
| Squeezenet_v1_0 | force_float16 | 1 | 0.348348 | 0.583584 | MLU370 X4 |
| Squeezenet_v1_0 | qint8_mixed_float16 | 1 | 0.355355  | 0.578579 | MLU370 X4 |

## 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* squeezenet_v1_0 caffemodel file下载链接：https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
* squeezenet_v1_0 prototxt file下载链接：https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/deploy.prototxt
* LSVRC_2012验证集链接: https://image-net.org/challenges/LSVRC
## 7.Release_Notes
@TODO
