# OpenPose Caffe

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用将 OpenPose 网络的 Caffe 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上，支持 BODY_25 及 COCO 两个模型。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [下载数据集和模型](#33-下载数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41gen_model-高级说明)
  - [infer_cpp 高级说明](#42infer_cpp-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 OpenPose 实现来自 github 开源项目https://github.com/CMU-Perceptual-Computing-Lab/openpose

下面将展示如何将该项目中Caffe实现的 OpenPose 模型转换为MagicMind的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
# git clone 本仓库
cd magicmind_cloud/buildin/cv/other/openpose_caffe
```

在开始运行代码前需要先安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```
另外，如果您之前未安装过`rapidjson`，则您还需安装`rapidjson`:
```bash
sudo apt-get update -y
sudo apt-get install -y rapidjson-dev
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `COCO_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 下载数据集和模型

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

```bash
cd ${PROJ_ROOT_PATH}/infer_cpp
#bash run.sh <magicmind_model>  <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000
```

使用 COCO API 计算精度:

```bash
python ${UTILS_PATH}/compute_coco_keypoints.py --ann_file ${COCO_DATASETS_PATH}/annotations/person_keypoints_val2017.json \
  --res_file ${PROJ_ROOT_PATH}/data/images/coco_force_float32_1/COCO \
  --output_file ${PROJ_ROOT_PATH}/data/images/force_float32_1_eval
```

结果：

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.155
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.127
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.113
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.155
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.142

```
**注意**：以上结果仅供示例用，实际求取精度时会使用COCO数据集的全部数据，此处仅使用了1000张图片。

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/other/openpose_caffe && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

OpenPose Caffe 模型转换为 OpenPose MagicMind 模型分成以下几步：

- 使用 MagicMind Parser 模块将 Caffe 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

**注意：**
在`gen_model.py`内使用了一些公共的组件，例如arg解析、第三方框架（如Caffe）模型解析、MagicMind 配置设定等，这些公共组件以及公共参数如`batch_size`, `device_id`的说明详见：[python公共组件的README.md](../../../python_common/README.md)

大部分参数为公共参数，网络特定参数如下：
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行推理任务。

### 4.2 infer_cpp 高级说明

概述：
本例使用 MagicMind C++ API 编写了名为 infer_cpp 的模型推理程序。其中程序主要由以下内容构成:

- infer.cpp: 高效率地将 MagicMind 模型运行在 MLU 板卡上。
- pre_precess.hpp, pre_precess.cpp: 前处理。
- post_precess.hpp, post_precess.cpp: 后处理。

参数说明:

- `magicmind_model`: MagicMind 模型路径。
- `image_dir`: 输入图像目录，程序对该目录下所有后缀为 jpg 的图片执行目标检测任务。
- `image_list`: 数据集文件列表文件。
- `save_img`: 结果可视化。若指定为 true，则保存渲染后的图像，默认为 false。
- `output_dir`: 根据检测结果进行渲染后的图像或 COCO API 风格检测结果文件保存路径。
- `network`: 指定推理模型是 BODY_25 还是 COCO，默认是COCO。该参数在`net_params.cpp`内定义。

**注意：**
在搭建`infer_cpp`时同样使用了一些公共组件/函数，如 MagicMind模型运行所需的上下文创建、获取模型输入输出维度、资源回收函数等，相关说明详见[cpp公共组件的README.md](../../../cpp_common/README.md)

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${magicmind_model} --batch_size ${batch_size} --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

通过快速使用中 3.5 的脚本跑出 OpenPose 在 COCO2017 数据集上的 AP 如下：
| Model   | Precision          | Batch_Size | AP IoU=0.50 |
| ------- | ------------------- | ---------- | ----------- |
| BODY_25 | force_float32       | 1          | 0.779       |
| BODY_25 | force_float16       | 1          | 0.780       |
| BODY_25 | qint8_mixed_float16 | 1          | 0.767       |
| COCO    | force_float32       | 1          | 0.752       |
| COCO    | force_float16       | 1          | 0.752       |
| COCO    | qint8_mixed_float16 | 1          | 0.743       |

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip
- COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- BODY_25 prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt
- BODY_25 caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel
- COCO prototxt 下载链接：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
- COCO caffemodel 下载链接：http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
- openpose GITHUB 下载链接：https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
- rapidjson 代码下载链接：https://github.com/miloyip/rapidjson.git

