# YOLOv5_PyTorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 yolov5 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1-模型概述)
- [前提条件](#2-前提条件)
- [快速使用](#3-快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4-高级说明)
  - [export_model 高级说明](#41-export_model-高级说明)
  - [gen_model 高级说明](#42-gen_model-高级说明)
  - [infer_cpp 高级说明](#43-infer_cpp-高级说明)
  - [infer_python 高级说明](#44-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 yolov5 实现来自 github 开源项目[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 中的 v6.1 版本。下面将展示如何将该项目中 PyTorch 实现的 yolov5 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/yolov5_v6_1_pytorch
```

在开始运行代码前需要执行以下命令安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `COCO_DATASETS_PATH`, 并且执行以下命令：

```bash
# for instance, COCO_DATASETS_PATH is : /home/your_path/coco_datasets
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd ${PROJ_ROOT_PATH}/gen_model
# bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
# 指定您想输出的magicmind_model路径，例如./model
bash run.sh ${magicmind_model} force_float32 1 true
```

### 3.5 执行推理

1. infer_cpp 执行推理

```bash
cd ${PROJ_ROOT_PATH}/infer_cpp
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000 
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.668
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.789
```

2. infer_python

```bash
cd ${PROJ_ROOT_PATH}/infer_python
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000 
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.668
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.789
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/detection/yolov5_v6_1_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 export_model 高级说明

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
patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < ${PROJ_ROOT_PATH}/export_model/init.patch
patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < ${PROJ_ROOT_PATH}/export_model/activation.patch
```

2.为了提高性能，需要将原 yolov5 模型中的 detect 层后处理去掉,生成修改后的 pt,之后再在生成 MagicMind Model 时添加 yolov5 后处理算子。
2.1 去掉 yolov5 网络最后的 detect layer（此 demo 的 detect 部分会放在 MLU 上运行，具体由目录下文件夹 gen_model 中 python 代码实现）修改如下:

```bash
cd ${PROJ_ROOT_PATH}/export_model/yolov5
# 修改yolov5/models/yolo.py, yolov5/models/experimental.py 以及 yolov5/export.py
git apply ${PROJ_ROOT_PATH}/export_model/yolov5_v6.1_pytorch.patch
```

2.2 使用下面的代码导出 jit.trace 模型文件。

```bash
python ${PROJ_ROOT_PATH}/export_model/yolov5/export.py --weights ${PROJ_ROOT_PATH}/data/models/yolov5m.pt --imgsz 640 640 --include torchscript --batch-size 1
```

### 4.2 gen_model 高级说明

PyTorch yolov5 模型转换为 MagicMind yolov5 模型分成以下几步：

- 使用 MagicMind Parser 模块将 torch.jit.trace 生成的 pt 文件解析为 MagicMind 网络结构。
- 模型量化。
- 添加 yolov5 后处理算子。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `conf_thres`: confidence_thresh，检测框得分阈值。
- `iou_thres`: nms_thresh。
- `max_det`: limit_detections。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.3 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 yolov5 目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
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

### 4.4 infer_python 高级说明

参数说明：

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_num`: 输入图像数量。
- `file_list`: 数据集文件列表文件。
- `label_path`: 标签文件路径。
- `result_file`: COCO API 风格检测结果文件保存路径。
- `result_img`: 根据检测结果进行渲染后的图像。
- `save`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}
bash benchmark/eval.sh
```

一键运行 benchmark 里的脚本跑出 YOLOv5m 在 COCO2017 数据集中 MAP 如下：

| Model  | Precision           | Batch_Size | mAP (0.5:0.95) | mAP (0.5) |
| ------ | ------------------- | ---------- | -------------- | --------- |
| YOLOv5m | force_float32       | 1          | 0.451          | 0.639     |
| YOLOv5m | force_float16       | 1          | 0.450          | 0.639     |
| YOLOv5m | qint8_mixed_float16 | 1          | 0.425          | 0.621     |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：[http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- COCO VAL2017 标签下载链接：[http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- YOLOV5M 模型下载链接：[https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt)
- YOLOV5 GITHUB 下载链接：[https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)


