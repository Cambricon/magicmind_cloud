# yolov3_caffe

MagicMind是面向寒武纪MLU(Machine Learning Unit,机器学习单元)的推理加速引擎。MagicMind能将深度学习框架(Tensorflow,PyTorch,ONNX等) 
训练好的算法模型转换成MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本sample探讨如何使用将yolov3网络的Caffe模型转换为MagicMind模型，进而部署在寒武纪MLU板卡上。

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

 本例使用的yolov3实现来自github开源项目https://github.com/pjreddie/darknet。 下面将展示如何将该项目中Caffe实现的yolov3模型转换为MagicMind的模型。

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
- 下载模型
将darknet原生的yolov3.cfg和yolv3.weight转换为本仓库所需要的yolov3.caffemodel和yolov3.prototxt，请参考[这里](http://gitlab.software.cambricon.com/neuware/software/solutionsdk/caffe_yolo_magicmind)。


### 3.4编译MagicMind模型
```bash
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 false 1
```

结果：

```bash 
Generate model done, model save to yolov3_caffe/data/models/yolov3_caffe_model_force_float32_false_1
```

### 3.5执行推理
1.infer_cpp
```bash 
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh force_float32 false 1
```
计算精度:
```
python $UTILS_PATH/compute_coco_mAP.py  --file_list $PROJ_ROOT_PATH/data/output/force_float32_false_1/json/image_name.txt \
                                        --result_dir$PROJ_ROOT_PATH/data/output/force_float32_false_1/results \
                                        --ann_dir $DATASETS_PATH/ \
                                        --data_type 'val2017' \
                                        --json_name$PROJ_ROOT_PATH/data/output/force_float32_false_1/json/force_float32_false_1 \
                                        --img_dir $DATASETS_PATH/val2017 \
                                        --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/force_float32_false_1_log_eval
```
结果:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.681
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
```

### 3.6一键运行
以上3.3~3.5的步骤也可以通过运行./run.sh来实现一键执行

## 4.高级说明
### 4.1 gen_model细节说明
Caffe yolov3模型转换为MagicMind yolov3模型分成以下几步：
* 使用MagicMind Parser模块将caffe文件解析为MagicMind网络结构。
* 模型量化。
* 使用MagicMind Builder模块生成MagicMind模型实例并保存为离线模型文件。

参数说明:
* `CAFFEMODEL`: yolov3 caffe的权重路径。
* `PROTOTXT`: yolov3 caffe的网络结构路径。
* `MM_MODEL`: 保存MagicMind模型路径。
* `DATASET_DIR`: 校准数据文件路径。
* `QUANT_MODE`: 量化模式，如forced_float32，forced_float16，qint8_mixed_float16。
* `SHAPE_MUTABLE`: 是否生成可变batch_size的MagicMind模型。
* `BATCH_SIZE`: 生成可变模型时batch_size可以随意取值，生成不可变模型时batch_size的取值需要对应pt的输入维度。
* `DEV_ID`: 设备号。

### 4.2 infer_cpp细节说明
概述：
本例使用MagicMind C++ API编写了名为infer_cpp的视频检测程序。infer_cpp将展示如何使用MagicMind C++ API构建高效的yolov3目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:
* infer.cpp: 高效率的将MagicMind模型运行在MLU板卡上。

参数说明:
* device_id: MLU设备号
* batch_size: 模型batch_size
* magicmind_model: MagicMind模型路径。
* image_dir: 数据集路径
* label_path：coco.names文件
* output_img_dir:推理输出-画框图像路径
* output_pred_dir：推理输出-结果文件路径
* save_imgname_dir：推理输出-所有经过推理的图像名称会被放置于一个名称为image_name.txt文件当中，用于精度验证。
* save_img：是否存储推理输出画框图像 1 存储 0 不存储
* save_pred:是否存储推理结果txt文件 1 存储 0 不存储
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
| yolov3 | forced_float32_false_1  | 42.6006 | 23.461 | 24.093 | 24.917  | MLU370 S4 |
| yolov3 | forced_float32_false_4  | 89.5774 | 44.641 | 49.468 | 51.208  | MLU370 S4 |
| yolov3 | forced_float32_false_8  | 84.914 | 68.303 | 76.27 | 78.331  | MLU370 S4 |
| yolov3 | forced_float16_false_1 | 116.141 | 8.5954 |  8.995  | 9.26     |  MLU370 S4  |
| yolov3 | forced_float16_false_4 | 281.059 | 14.22   | 16.308 | 17.218  |  MLU370 S4 |
| yolov3 | forced_float16_false_8 | 291.728 | 22.068 | 26.436 | 28.43  |  MLU370 S4 |
| yolov3 | qint8_mixed_float16_false_1 | 255.246 | 3.9066 | 3.91  | 3.912   | MLU370 S4  |
| yolov3 | qint8_mixed_float16_false_4 | 841.498 | 4.7415  | 4.749  | 4.752  |  MLU370 S4 |
| yolov3 | qint8_mixed_float16_false_8 | 1132.1 | 5.2873 |  5.53 |  5.67  |  MLU370 S4 |
| yolov3 | forced_float32_false_1  | 19.7373 | 50.653 | 132.29 | 56.19  | MLU370 X4 |
| yolov3 | forced_float32_false_4  | 61.8691|  64.64  | 163.03 | 258.67   | MLU370 X4 |
| yolov3 | forced_float32_false_8  | 74.1679| 83.126|209.78| 292.28  | MLU370 X4 |
| yolov3 | forced_float16_false_1 |  52.8324 | 18.915 |  57.051  | 113.01  |  MLU370 X4  |
| yolov3 | forced_float16_false_4 | 169.098 |23.643   | 80.003 | 121.58 |  MLU370 X4 |
| yolov3 | forced_float16_false_8 | 219.786 | 27.287 | 98.417 | 142.24  |  MLU370 X4 |
| yolov3 | qint8_mixed_float16_false_4 | 139.524 | 7.1393  | 15.518  | 59.597  |  MLU370 X4 |
| yolov3 | qint8_mixed_float16_false_1 | 347.624 | 11.493| 33.22  | 91.624   | MLU370 X4  |
| yolov3 | qint8_mixed_float16_false_8 | 515.365 | 11.86 |  39.141 |   97.038  |  MLU370 X4 |

### 5.2精度benchmark结果
一键运行benchmark里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```
通过快速使用中3.6的脚本跑出yolov3在coco val2017数据集上的mAP如下：**测试结果表明yolov3在S4上与X4结果一致**
| Model  | QuantMode_ShapeMutable_BatchSize | mAP(0.50:0.95) |  mAP(0.50) |MLU板卡类型 |
| --------- | ---------- | ---------- | --------- | ---------
| yolov3 | force_float32_false_1 | 0.378|  0.674 | MLU370 S4 |
| yolov3 | force_float32_false_4 | 0.378|  0.674 | MLU370 S4 |
| yolov3 | force_float32_false_8 | 0.378|  0.674 | MLU370 S4 |
| yolov3 | force_float16_false_1 | 0.379|  0.674 | MLU370 S4 |
| yolov3 | force_float16_false_4 | 0.379|  0.674 | MLU370 S4 |
| yolov3 | force_float16_false_8 | 0.379|  0.674 | MLU370 S4 |
| yolov3 | qint8_mixed_float16_false_1 | 0.350|  0.653 | MLU370 S4 |
| yolov3 | qint8_mixed_float16_false_4 | 0.350|  0.652 | MLU370 S4 |
| yolov3 | qint8_mixed_float16_false_8 | 0.350|  0.653 | MLU370 S4 |
| yolov3 | force_float32_false_1 | 0.378|  0.674 | MLU370 X4 |
| yolov3 | force_float32_false_4 | 0.378|  0.674 | MLU370 X4 |
| yolov3 | force_float32_false_8 | 0.378|  0.674 | MLU370 X4 |
| yolov3 | force_float16_false_1 | 0.379|  0.674 | MLU370 X4 |
| yolov3 | force_float16_false_4 | 0.379|  0.674 | MLU370 X4 |
| yolov3 | force_float16_false_8 | 0.379|  0.674 | MLU370 X4 |
| yolov3 | qint8_mixed_float16_false_1 | 0.350|  0.653 | MLU370 X4 |
| yolov3 | qint8_mixed_float16_false_4 | 0.350|  0.653 | MLU370 X4 |
| yolov3 | qint8_mixed_float16_false_8 | 0.350|  0.653 | MLU370 X4 |

## 6.免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
* yolov3 caffemodel file下载链接：https://www.dropbox.com/s/bf5z2jw1pg07c9n/yolov3_resnet18_ucf101_r2_ft_iter_20000.caffemodel?dl=0
* yolov3 prototxt file下载链接：https://raw.githubusercontent.com/xiaoqi25478/network_resources/main/yolov3.prototxt
* coco数据集下载链接： http://images.cocodataset.org/zips/val2017.zip
## 7.Release_Notes
@TODO
