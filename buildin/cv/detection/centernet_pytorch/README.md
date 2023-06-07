# centernet_pytorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 CenterNet 网络的 PyTorch 实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [模型转换](#34-模型转换)
  - [编译 MagicMind 模型](#35-编译-magicmind-模型)
  - [执行推理](#36-执行推理)
  - [一键运行](#37-一键运行)
- [高级说明](#4高级说明)
  - [export_model 高级说明](#41-export_model-高级说明)
  - [gen_model 高级说明](#42-gen_model-高级说明)
  - [infer_cpp 高级说明](#43-infer_cpp-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)


## 1.模型概述

本例使用的 CenterNet 实现来自 github 开源项目[https://github.com/xingyizhou/CenterNet/tree/2b7692c377c6686fb35e473dac2de6105eed62c6](https://github.com/xingyizhou/CenterNet/tree/2b7692c377c6686fb35e473dac2de6105eed62c6)。

下面将展示如何将该项目中 PyTorch 实现的 CenterNet 网络转换为 MagicMind 的模型。由于 MagicMind 暂不支持 DCN 算子，本例可使用的 Backbone 有 Hourglass 以及 DLAV0，下面使用 DLAV0 作为 backbone 进行部署。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`


### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/detection/centernet_pytorch
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `COCO_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd $PROJ_ROOT_PATH/export_model
bash get_datasets_and_models.sh
```

### 3.4 模型转换

```bash
cd $PROJ_ROOT_PATH/export_model
#bash run.sh <batch_size>
bash run.sh 1
```

### 3.5 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
# bash run.sh <magicmind_model> <precision> <batch_size> <shape_mutable> 
bash run.sh ../data/models/centernet_pytorch_model_qint8_mixed_float16_true qint8_mixed_float16 1 true 
```

结果：

```bash
Generate model done, model save to /mm_ws/proj/modelzoo/magicmind_cloud/buildin/cv/detection/centernet_pytorch/data/models/centernet_pytorch_model_qint8_mixed_float16_true
```

### 3.6 执行推理

1. infer_cpp 
```bash
cd $PROJ_ROOT_PATH/infer_cpp
# bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ../data/models/centernet_pytorch_model_qint8_mixed_float16_true 1  10
```


结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.282
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.330
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764

```

### 3.7 一键运行

以上 3.3~3.6 的步骤也可以通过运行cd modelzoo/magicmind_cloud/buildin/cv/detection/centernet_pytorch && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 export_model 高级说明

向 MagicMind 导入 PyTorch 模型首先需要先使用 torch.jit.trace 生成 PyTorch 模型文件。

首先，执行以下代码块修改 CenterNet 源码。将 CenterNet 后处理过程中 Heatmap 的池化操作放入网络中执行，这样在后面的检测程序中，池化操作将包含在 MagicMind 模型中，并运行在 MLU 上。
加入此项修改后，网络变为 4 个输出: hm_max(经过池化后的 heatmap)，heatmap，wh(检测框宽高), reg(回归中心点偏移量)。

参考https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/decode.py 中\_nms 函数实现。

```bash
cd $PROJ_ROOT_PATH/export_model
# 这个patch文件只修改了dlav0作为backbone的实现，若需使用其它backbone，可参照centernet.diff文件中的修改内容。
patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/networks/dlav0.py < centernet_dlav0.diff
# 注释掉DCN相关内容
patch CenterNet-2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/model.py < centernet_model.diff
# 使用$PROJ_ROOT_PATH/export_model/export.py jit.trace导出模型。
python $PROJ_ROOT_PATH/export_model/export.py --model_weight $MODEL_PATH/ctdet_coco_dlav0_1x.pth \
    					      --input_width 512 \
    					      --input_height 512 \
    					      --batch_size 1 \
    					      --traced_pt $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_1bs.pt
```

### 4.2 gen_model 高级说明

PyTorch CenterNet 网络转换为 MagicMind 模型可分为以下几个步骤: 1.通过 torch.jit.trace 生成 pt 文件。 2.使用 MagicMind Parser 模块将 torch.jit.trace 生成的 pt 文件解析为 MagicMind 网络结构。 3.模型量化(可选)。 4.使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pt_model`: 转换后 pt 的路径。
- `output_model`: 保存 MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `precision`: 精度模式，如 force_float32，force_float16，qint8_mixed_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以在dim range范围内取值，生成不可变模型时 batch_size 的取值需要对应 pt 的输入维度。

### 4.3 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的目标检测程序。infer_cpp 将展示如何使用 MagicMind C++ API 构建高效的 centernet 目标检测(图像预处理=>推理=>后处理)。其中程序主要由以下内容构成:

- infer.hpp, infer.cpp: 高效率的将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。
- post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_num`: 输入图像数量。
- `file_list`: 数据集文件列表文件。
- `label_path`: 标签文件路径。
- `max_bbox_num`:最大毛框数量。
- `confidence_thresholds`: confidence_thresh，检测框得分阈值。
- `output_dir`: 根据检测结果进行渲染后的图像或 COCO API 风格检测结果文件保存路径。
- `save_img`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。
- `batch_size`: 推理时，输入 tensor 的维度

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd $PROJ_ROOT_PATH
bash benchmark/eval.sh
```

通过快速使用中 3.6 的脚本跑出 centernet 在 COCO2017 数据集上的 mAP 如下：
| Model | Precision | Batch_Size | mAP (0.5:0.95) | mAP (0.5) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| CenterNet | force_float32 | 1 | 0.214 | 0.367 |
| CenterNet | force_float16 | 1 | 0.214 | 0.366 |
| CenterNet | qint8_mixed_float16 | 1 | 0.183 | 0.329 |


## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：[http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- COCO VAL2017 标签下载链接：[http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- Centernet 权重文件下载链接：[https://drive.google.com/uc?id=18yBxWOlhTo32_swSug_HM4q3BeWgxp_N](https://drive.google.com/uc?id=18yBxWOlhTo32_swSug_HM4q3BeWgxp_N)
- CenterNet 实现源码下载链接：[https://github.com/xingyizhou/CenterNet/archive/2b7692c377c6686fb35e473dac2de6105eed62c6.zip](https://github.com/xingyizhou/CenterNet/archive/2b7692c377c6686fb35e473dac2de6105eed62c6.zip)


