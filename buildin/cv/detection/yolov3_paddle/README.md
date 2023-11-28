# yolov3_paddle

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何将 Paddle_YOLOv3 网络的 paddle 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。
/
## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-MagicMind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 代码解释](#41-gen_model-代码解释)
  - [infer_python 代码解释](#42-infer_python-代码解释)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 paddle 实现来自 github 开源项目[https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3)。 下面将展示如何将该项目中 paddle 实现的 yolov3 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/yolov3_paddle
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

- 下载数据集

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size
bash run.sh force_float32 false 1

```

结果：

```bash
Generate model done, model save to yolov3_paddle/data/models/yolov3_paddle_model_force_float32_false_1
```

### 3.5 执行推理

1.infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision shape_mutable batch_size image_num
bash run.sh force_float32 false 1 5000
```

计算精度:

```
python $UTILS_PATH/compute_coco_mAP.py  --file_list $PROJ_ROOT_PATH/data/output/force_float32_false_1/json/image_name.txt \
                                        --result_dir$PROJ_ROOT_PATH/data/output/force_float32_false_1/results \
                                        --ann_dir $COCO_DATASETS_PATH/ \
                                        --data_type 'val2017' \
                                        --json_name$PROJ_ROOT_PATH/data/output/force_float32_false_1/json/force_float32_false_1 \
                                        --img_dir $COCO_DATASETS_PATH/val2017 \
                                        --image_num 5000 2>&1 | tee $PROJ_ROOT_PATH/data/output/force_float32_false_1_eval
```

结果:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638

```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/detection/yolov3_paddle && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

Paddle yolov3 模型转换为 MagicMind yolov3 模型分成以下几步：

- 使用 MagicMind Parser 模块将 paddle 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `ONNXMODEL`: Paddle_YOLOv3 onnx 的权重路径。
- `MM_MODEL`: 保存 MagicMind 模型路径。
- `DATASET_DIR`: 校准数据文件路径。
- `PRECISION`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `SHAPE_MUTABLE`: 是否生成可变 batch_size 的 MagicMind 模型。
- `BATCH_SIZE`: 生成可变模型时 batch_size 可以在 dimension range 范围内取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `DEV_ID`: 设备号。

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind python API 编写了名为 infer_python的视频检测程序。infer_python 将展示如何使用 MagicMind python API 构建高效的 Paddle_YOLOv3 目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:

- infer_python: 高效率地将 MagicMind 模型运行在 MLU 板卡上。

参数说明:

- device_id: MLU 设备号
- batch_size: 模型 batch_size
- magicmind_model: MagicMind 模型路径。
- image_dir: 数据集路径
- label_path：coco.names 文件
- output_img_dir:推理输出-画框图像路径
- output_pred_dir：推理输出-结果文件路径
- save_imgname_dir：推理输出-所有经过推理的图像名称会被放置于一个名称为 image_name.txt 文件当中，用于精度验证。
- save_img：是否存储推理输出画框图像 1 存储 0 不存储
- save_pred:是否存储推理结果 txt 文件 1 存储 0 不存储
- test_nums: 输入数据数量 默认-1 表示检查全部输入数据

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh
```
通过快速使用中 3.7 的脚本跑出 Paddle_YOLOv3 在 coco val2017 数据集上的 mAP 如下：**测试结果如下**
| Model | backbone | input size | recision_ShapeMutable_BatchSize | mAP(0.50:0.95) | mAP(0.50) |
| --------- | ---------- | ---------- | --------- |
| Paddle_YOLOv3 | DarkNet53 | 608 | force_float32_false_1 | 0.391| 0.619 |
| Paddle_YOLOv3 | DarkNet53 | 608 | force_float16_false_1 | 0.391| 0.619 |
| Paddle_YOLOv3 | DarkNet53 | 608 | qint8_mixed_float16_false_1 | 0.308| 0.520 |
## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- coco 数据集下载链接： http://images.cocodataset.org/zips/val2017.zip
- paddle yolov3:  https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3
