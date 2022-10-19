# yolov5_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 yolov5 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1.模型概述)
- [前提条件](#2.前提条件)
- [快速使用](#3.快速使用)
  - [环境准备](#3.1环境准备)
  - [下载仓库](#3.2下载仓库)
  - [下载数据集，模型](#3.3下载数据集,模型)
  - [模型转换](#3.4模型转换)
  - [编译 MagicMind 模型](#3.5编译MagicMind模型)
  - [执行推理](#3.6执行推理)
  - [一键运行](#3.7一键运行)
- [高级说明](#4.高级说明)
  - [export_model 高级说明](#4.1export_model高级说明)
  - [gen_model 高级说明](#4.2gen_model高级说明)
  - [infer_cpp 高级说明](#4.3infer_cpp高级说明)
  - [infer_python 高级说明](#4.3infer_python高级说明)
- [精度和性能 benchmark](#5.精度和性能benchmark)
  - [性能 benchmark 结果](#5.1性能benchmark结果)
  - [精度 benchmark 结果](#5.2精度benchmark结果)
- [免责声明](#6.免责声明)
- [Release notes](#7.Release_Notes)

## 1.模型概述

本例使用的 yolov5 实现来自 github 开源项目https://github.com/ultralytics/yolov5 中的 v6.1 版本。下面将展示如何将该项目中 PyTorch 实现的 yolov5 模型转换为 MagicMind 的模型。

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

在开始运行代码前需要执行以下命令安装必要的库：

```bash
pip install -r requirement.txt
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集,模型

```bash
cd $PROJ_ROOT_PATH/export_model
./get_datasets_and_models.sh
```

### 3.4 模型转换

```bash
cd $PROJ_ROOT_PATH/export_model
./run.sh
```

### 3.5 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
./run.sh qint8_mixed_float16 true 1 0.001 0.65 1000
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/detection/yolov5_v6_1_pytorch/data/models/yolov5_pytorch_model_qint8_mixed_float16_true_1
```

### 3.6 执行推理

1. infer_cpp 执行推理
   编译 infer_cpp 目录，在当前目录输出 x86 可执行文件`infer`:

```bash
cd $PROJ_ROOT_PATH/infer_cpp
./build.sh
```

推理：

```bash
./run.sh
```

使用 COCO API 计算精度:

```bash
python $UTILS_PATH/compute_coco_mAP.py --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                       --result_dir $PROJ_ROOT_PATH/data/images/infer_cpp_output_dir_qint8_mixed_float16_true_1 \
                                       --ann_dir $DATASETS_PATH \
                                       --data_type val2017 \
                                       --json_name $PROJ_ROOT_PATH/data/json/yolov5_qint8_mixed_float16_true_1 \
                                       --img_num 10
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.495
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.690
```

2. infer_python

```bash
cd $PROJ_ROOT_PATH/infer_python
./run.sh qint8_mixed_float16 true 1 10
```

### 3.7 一键运行

以上 3.3~3.6 的步骤也可以通过运行./run.sh 来实现一键执行

## 4.高级说明

### 4.1export_model 高级说明

1.由于 magicmind 最高支持 pytorch 1.6.0 版本，此版本没有 SiLU 函数，所以要在环境中修改代码如下：

```bash
vim /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py
#添加如下函数定义
class SiLU(Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

vim /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py
#添加声明
from .activation import SiLU
__all__ = [ *, 'SiLU']
```

或者直接运行以下代码：

```bash
patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < $PROJ_ROOT_PATH/export_model/init.patch
patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < $PROJ_ROOT_PATH/export_model/activation.patch
```

2.为了提高性能，需要将原 yolov5 模型中的 detect 层后处理去掉,生成修改后的 pt,之后再在生成 MagicMind Model 时添加 yolov5 后处理算子。
2.1 去掉 yolov5 网络最后的 detect layer（此 demo 的 detect 部分会放在 MLU 上运行，具体由目录下文件夹 gen_model 中 python 代码实现）修改如下:

```bash
cd $PROJ_ROOT_PATH/export_model/yolov5
# 修改yolov5/models/yolo.py, yolov5/models/experimental.py 以及 yolov5/export.py
git apply $PROJ_ROOT_PATH/export_model/yolov5_v6.1_pytorch.patch
```

2.2 使用下面的代码导出 jit.trace 模型文件。

```bash
python $PROJ_ROOT_PATH/export_model/yolov5/export.py --weights $PROJ_ROOT_PATH/data/models/yolov5m.pt --imgsz 640 640 --include torchscript --batch-size 1
```

### 4.2gen_model 高级说明

PyTorch yolov5 模型转换为 MagicMind yolov5 模型分成以下几步：

- 使用 MagicMind Parser 模块将 torch.jit.trace 生成的 pt 文件解析为 MagicMind 网络结构。
- 模型量化。
- 添加 yolov5 后处理算子。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: 转换后 pt 的路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `output_model_path`: 保存 MagicMind 模型路径。
- `quant_mode`: 量化模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以随意取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。
- `conf_thres`: confidence_thresh，检测框得分阈值。
- `iou_thres`: nms_thresh。
- `max_det`: limit_detections。

### 4.3infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 yolov5 目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.hpp, infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。
- post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_num`: 输入图像数量。
- `file_list`: 数据集文件列表文件。
- `label_path`: 标签文件路径。
- `output_dir`: 根据检测结果进行渲染后的图像或 COCO API 风格检测结果文件保存路径。
- `save`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。
- `batch_size`: 推理时，输入 tensor 的维度

### 4.4infer_python 高级说明

参数说明：

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_num`: 输入图像数量。
- `file_list`: 数据集文件列表文件。
- `label_path`: 标签文件路径。
- `result_file`: COCO API 风格检测结果文件保存路径。
- `result_img`: 根据检测结果进行渲染后的图像。
- `save`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。

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
cd $PROJ_ROOT_PATH/benchmark
./perf.sh
```

### 5.2 精度 benchmark 结果

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH/benchmark
./eval.sh
```

通过快速使用中 3.6 的脚本跑出 yolov5s 在 COCO2017 数据集上的 mAP 如下：
| Model | Quant_Mode | Batch_Size | mAP (0.5:0.95) | mAP (0.5) |MLU 板卡类型 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| YoloV5 | force_float32 | 1 | 0.363 | 0.529 | MLU370 S4 |
| YoloV5 | force_float16 | 1 | 0.366 | 0.530 | MLU370 S4 |
| YoloV5 | qint8_mixed_float16 | 1 | 0.348 | 0.520 | MLU370 S4 |

| Model  | Quant_Mode          | Batch_Size | mAP (0.5:0.95) | mAP (0.5) | MLU 板卡类型 |
| ------ | ------------------- | ---------- | -------------- | --------- | ------------ |
| YoloV5 | force_float32       | 1          | 0.363          | 0.529     | MLU370 X4    |
| YoloV5 | force_float16       | 1          | 0.366          | 0.530     | MLU370 X4    |
| YoloV5 | qint8_mixed_float16 | 1          | 0.348          | 0.520     | MLU370 X4    |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
- COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- YOLOV5M 模型下载链接：https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt
- YOLOV5 GITHUB 下载链接：https://github.com/ultralytics/yolov5.git

## 7.Release_Notes

@TODO
