# mobilenetv3_pytorch

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX等) 
训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本sample探讨如何使用将mobilenetv3网络的pytorch模型转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

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

 本例使用的mobilenetv3实现来自github开源项目https://github.com/kuan-wang/pytorch-mobilenet-v3。 下面将展示如何将该项目中pytorch实现的mobilenetv3模型转换为MagicMind的模型。

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
- 下载数据集
```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4编译MagicMind模型
```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1
```

结果：

```bash 
Generate model done, model save to mobilenetv3_pytorch/data/mm_model/force_float32_false_1
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
结果:
```
top1 accuracy: 0.658317
top5 accuracy: 0.876754
```

### 3.6一键运行
以上3.3~3.5的步骤也可以通过运行./run.sh来实现一键执行

## 4.高级说明
### 4.1 gen_model细节说明
pytorch mobilenetv3模型转换为MagicMind mobilenetv3模型分成以下几步：
* 使用MagicMind Parser模块将pytorch文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `PT_MODEL`: mobilenetv3 pytorch的权重路径。
* `MM_MODEL`: 保存MagicMind模型路径。
* `DATASET_DIR`: 校准数据文件路径。
* `QUANT_MODE`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `SHAPE_MUTABLE`: 是否生成可变batch_size的MagicMind模型。
* `BATCH_SIZE`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。
* `DEV_ID`: 设备号。

### 4.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的mobilenetv3目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。

参数说明:
* device_id: MLU设备号
* batch_size: 模型batch_size
* magicmind_model: MagicMind模型路径。
* image_dir: 数据集路径
* label_file:ground truth文件
* output_dir:推理输出-画框图像路径
* result_file:推理结果输出文件 txt格式
* result_label_file 推理结果输出label文件 txt格式
* result_top1_file:top1推理结果输出label文件 txt格式
* result_top5_file:top5推理结果输出label文件 txt格式
* image_num:推理图片数量
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
| Model | QuantMode_ShapeMutable_BatchSize | Throughput (qps) | MLU compute Latency Avg (ms) | 95% (ms) | 99% (ms) | MLU板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| mobilenetv3 | forced_float32_false_1  | 765.576 | 1.294 | 1.296  |  1.308   | MLU370 S4 |
| mobilenetv3 | forced_float32_false_4  | 2610.16 |  1.5195  | 1.527| 1.541  | MLU370 S4 |
| mobilenetv3 | forced_float32_false_8  |  3599.61 | 2.2105|2.217 | 2.22  | MLU370 S4 |
| mobilenetv3 | forced_float16_false_1 |  884.126 | 1.1144 |  1.118   |1.128 |  MLU370 S4  |
| mobilenetv3 | forced_float16_false_4 | 3021.6| 1.3118    | 1.317  |  1.319|  MLU370 S4 |
| mobilenetv3 | forced_float16_false_8 | 4288.74 | 1.8533 |1.859 | 1.862 |  MLU370 S4 |
| mobilenetv3 | qint8_mixed_float16_false_4 | 930.652| 1.0626  |  1.067  | 1.076     |  MLU370 S4 |
| mobilenetv3 | qint8_mixed_float16_false_1 | 3168.6| 1.2459  | 1.25    | 1.285 | MLU370 S4  |
| mobilenetv3 | qint8_mixed_float16_false_8 | 4444.96 | 1.7878 |  1.793 |   1.796   |  MLU370 S4 |
| mobilenetv3 | force_float32_false_1 |  758.914 | 1.3031   | 1.305  | 1.316 | MLU370 X4 |
| mobilenetv3 | force_float32_false_4 |  2463.13 | 1.6129 | 1.617 | 1.621  | MLU370 X4 |
| mobilenetv3 | force_float32_false_8 |  4214.74| 1.8878  | 1.894| 1.897  | MLU370 X4 |
| mobilenetv3 | force_float16_false_1 |  875.145 | 1.125 | 1.126   | 1.14  | MLU370 X4 |
| mobilenetv3 | force_float16_false_4 | 2848.21 | 1.3938    | 1.397 | 1.4   |  MLU370 X4 |
| mobilenetv3 | force_float16_false_8 |  5026.61 |  1.5704   | 1.573 | 1.579  |  MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_1 | 911.996 | 1.0657  | 1.069 | 1.084   |  MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_4 | 3018.75 | 1.3037  | 1.319  | 1.335  |  MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_8 | 5353.18 | 1.4812  | 1.489 | 1.566 |  MLU370 X4 |

### 5.2精度benchmark结果
一键运行benchmark里的脚本：
```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```
通过快速使用中3.6的脚本跑出mobilenetv3在imagenet1000数据集上的top1和top5精度如下
| Model  | QuantMode_ShapeMutable_BatchSize | top1 |  top5 |MLU板卡类型 |
| --------- | ---------- | ---------- | --------- | ---------|
| mobilenetv3 | force_float32_false_1 | 0.658|  0.876 | MLU370 S4 |
| mobilenetv3 | force_float32_false_4 | 0.658|  0.658| MLU370 S4 |
| mobilenetv3 | force_float32_false_8 | 0.657|  0.875 | MLU370 S4 |
| mobilenetv3 | force_float16_false_1 | 0.658|  0.876 | MLU370 S4 |
| mobilenetv3 | force_float16_false_4 | 0.658|  0.876 | MLU370 S4 |
| mobilenetv3 | force_float16_false_8 | 0.657|  0.875 | MLU370 S4 |
| mobilenetv3 | qint8_mixed_float16_false_1 | 0.653|  0.866 | MLU370 S4 |
| mobilenetv3 | qint8_mixed_float16_false_4 | 0.646|  0.862 | MLU370 S4 |
| mobilenetv3 | qint8_mixed_float16_false_8 | 0.645 |  0.867 | MLU370 S4 |
| mobilenetv3 | force_float32_false_1 | 0.658| 0.876 | MLU370 X4 |
| mobilenetv3 | force_float32_false_4 | 0.658|  0.876 | MLU370 X4 |
| mobilenetv3 | force_float32_false_8 | 0.657|  0.875 | MLU370 X4 |
| mobilenetv3 | force_float16_false_1 | 0.658|  0.876 | MLU370 X4 |
| mobilenetv3 | force_float16_false_4 | 0.658|  0.876 | MLU370 X4 |
| mobilenetv3 | force_float16_false_8 | 0.657|  0.875 | MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_1 | 0.653|  0.866 | MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_4 | 0.653|  0.862 | MLU370 X4 |
| mobilenetv3 | qint8_mixed_float16_false_8 | 0.6465|  0.862 | MLU370 X4 |

## 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* mobilenetv3 pth模型下载链接：https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C
## 7.Release_Notes
@TODO
