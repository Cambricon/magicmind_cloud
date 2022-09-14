# c3d_caffe

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX等) 
训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本sample探讨如何使用将c3d网络的Caffe实现转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

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
  * [gen_model代码解释](#4.1gen_model代码解释)
  * [infer_cpp代码解释](#4.2infer_cpp代码解释)
* [精度和性能benchmark](#5.精度和性能benchmark)
  * [性能benchmark结果](#5.1性能benchmark结果)
  * [精度benchmark结果](#5.2精度benchmark结果)
* [免责声明](#6.免责声明)
* [Release notes](#7.Release_Notes)

## 1.模型概述

 本例使用的c3d实现来自github开源项目https://github.com/facebookarchive/C3D/tree/master/C3D-v1.1。 下面将展示如何将该项目中Caffe实现的c3d模型转换为MagicMind的模型。

## 2.前提条件

* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本 MLU370 S4或 MLU370 X4，并安装好驱动(>=v4.20.6)；
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## 3.快速使用
### 3.1环境准备@TODO
```bash
下载MagicMind镜像："yellow.hub.cambricon.com/magicmind/daily/x86_64/magicmind:0.13.0-master-x86_64-ubuntu18.04-py_3_7"
docker load -i xxx.tar.gz
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
bash run.sh force_float32 false 1
```
```bash 
Generate model done, model save to c3d_caffe/data/models/c3d_caffe_model_force_float32_false_1
```

### 3.5执行推理
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


### 3.6一键运行
以上3.3~3.5的步骤也可以通过运行./run.sh来实现一键执行

## 4.高级说明
### 4.1 gen_model细节说明
Caffe c3d模型转换为MagicMind c3d模型分成以下几步：
* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `CAFFEMODEL`: c3d caffe的权重路径。
* `PROTOTXT`: c3d caffe的网络结构路径。
* `MM_MODEL`: 保存MagicMind模型路径。
* `DATASET_DIR`: 校准数据文件路径。
* `QUANT_MODE`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `SHAPE_MUTABLE`: 是否生成可变batch_size的MagicMind模型。
* `BATCH_SIZE`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。
* `DEV_ID`: 设备号。

### 4.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的c3d视频捕捉(视频帧采样=>推理=>后处理)。其中程序主要由以下内容构成:
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。

参数说明:
* resized_w: 预处理相关参数。指定图像预处理中缩放大小。
* resized_h: 预处理相关参数。指定图像预处理中缩放大小。
* magicmind_model: MagicMind模型路径。
* video_list: 输入视频列表文件，文件中每一行为一个视频文件路径。
* output_dir: 动作识别结果保存目录。每一个视频片段的top5识别结果将保存为一个txt文件。
* sampling_rate: 视频帧采样频率，默认为2，意味着每两帧采样一帧。
* clip_step: 截取视频片段的移动步长，默认为-1，若为-1，clip_step等于CLIP_LEN * sampling_rate。
例如视频帧按1，2，3，4，5，6编号，clip_step指定为2，则第一个视频片段从编号为1的视频帧开始，第二个视频片段从编号为3的视频帧开始。

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

| Model  | QuantMode_ShapeMutable_BatchSize |Throughput (qps) | MLU compute Latency Avg (ms) | 95% (ms) | 99% (ms) | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
| C3D | force_float32_false_1 |  26.4652 | 37.773 | 54.082 | 60.456  | MLU370 S4 |
| C3D | force_float32_false_4 |  71.2793 | 56.105 | 78.788 | 82.389  | MLU370 S4 |
| C3D | force_float32_false_8 |  100.814 | 79.34 | 101.92 | 105.91  | MLU370 S4 |
| C3D | force_float16_false_1 |  66.162 | 15.101 | 33.546  | 37.718  | MLU370 S4 |
| C3D | force_float16_false_4 | 192.492 | 20.767  | 43.44  | 47.214   |  MLU370 S4 |
| C3D | force_float16_false_8 | 291.771 | 27.405   | 50.877 | 55.092  |  MLU370 S4 |
| C3D | qint8_mixed_float16_false_1 | 252.96 | 3.9402  | 6.127 | 28.382  |  MLU370 S4 |
| C3D | qint8_mixed_float16_false_4 | 514.704 | 7.7583   | 25.211 | 34.051   |  MLU370 S4 |
| C3D | qint8_mixed_float16_false_8 | 671.644 | 11.898   | 14.984  | 38.591   |  MLU370 S4 |
| C3D | force_float32_false_1 |  30.0278 | 33.29  | 37.234 | 47.184  | MLU370 X4 |
| C3D | force_float32_false_4 |  91.3226 | 43.789 | 57.619 | 61.683  | MLU370 X4 |
| C3D | force_float32_false_8 |  145.804 | 54.855  | 70.214| 74.407  | MLU370 X4 |
| C3D | force_float16_false_1 |  80.3762 | 12.424 | 19.16   | 30.125  | MLU370 X4 |
| C3D | force_float16_false_4 | 261.985 | 15.253    | 20.272 | 31.963   |  MLU370 X4 |
| C3D | force_float16_false_8 |  427.534 |  18.7   | 28.095 | 38.22  |  MLU370 X4 |
| C3D | qint8_mixed_float16_false_1 | 322.206 | 2.9381  | 3.383 | 9.355   |  MLU370 X4 |
| C3D | qint8_mixed_float16_false_4 | 752.547 | 5.2258  | 7.167  | 18.346  |  MLU370 X4 |
| C3D | qint8_mixed_float16_false_8 | 1019.77 | 7.7997  | 11.537 | 27.926  |  MLU370 X4 |

### 5.2精度benchmark结果
一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```
通过快速使用中3.6的脚本跑出c3d在UCF101 testlist01数据集上的mAP如下：
| Model  | QuantMode_ShapeMutable_BatchSize | @Acc(Top 1) |  @Acc(Top 5) |MLU板卡类型 |
| --------- | ---------- | ---------- | --------- | ---------
| C3D | force_float32_false_1 | 0.805666|  0.995788 | MLU370 S4 |
| C3D | force_float32_false_4 | 0.809610|  0.995623 | MLU370 S4 |
| C3D | force_float32_false_8 | 0.805180|  0.995522 | MLU370 S4 |
| C3D | force_float16_false_1 | 0.805433|  0.995322 | MLU370 S4 |
| C3D | force_float16_false_4 | 0.805570|  0.995322 | MLU370 S4 |
| C3D | force_float16_false_8 | 0.806170|  0.995422 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_1 | 0.805371|  0.994343 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_4 | 0.802633|  0.994324 | MLU370 S4 |
| C3D | qint8_mixed_float16_false_8 | 0.802212|  0.996747 | MLU370 S4 |
| C3D | force_float32_false_1 | 0.805398|  0.996723 | MLU370 X4 |
| C3D | force_float32_false_4 | 0.806398|  0.996323 | MLU370 X4 |
| C3D | force_float32_false_8 | 0.804398|  0.992156 | MLU370 X4 |
| C3D | force_float16_false_1 | 0.802267|  0.998622 | MLU370 X4 |
| C3D | force_float16_false_4 | 0.800311|  0.998423 | MLU370 X4 |
| C3D | force_float16_false_8 | 0.800268|  0.992313 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_1 | 0.803371|  0.994234 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_4 | 0.805626|  0.996522 | MLU370 X4 |
| C3D | qint8_mixed_float16_false_8 | 0.806042|  0.993535 | MLU370 X4 |

## 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* c3d caffemodel file下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/c3d_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
* prototxt 下载链接: https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/c3d_resnet18_r2_ucf101.prototxt
* UCF101数据集下载链接： https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar
* UCF101数据集标签下载连接：https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/UCF101TrainTestSplits-RecognitionTask.zip
## 7.Release_Notes
@TODO
