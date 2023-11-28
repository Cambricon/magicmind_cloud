# MobileNet-v3_PyTorch

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等)
训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。
本 sample 探讨如何使用将 MobileNet-v3 网络的 PyTorch 模型转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

## 目录

- [模型概述](#1模型概述)
- [前提条件](#2前提条件)
- [快速使用](#3快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [准备数据集和模型](#33-准备数据集和模型)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41-gen_model-高级说明)
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6免责声明)

## 1.模型概述

本例使用的 

------------

 实现来自 github 开源项目https://github.com/kuan-wang/pytorch-mobilenet-v3 下面将展示如何将该项目中 PyTorch 实现的 MobileNet-v3-small 模型转换为 MagicMind 的模型。

## 2.前提条件

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/classification/mobilenetv3_pytorch
```
在开始运行代码前需要安装依赖，执行以下命令：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 env.sh 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `ILSVRC2012_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

- 下载数据集

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

1.infer_python

```bash
cd ${PROJ_ROOT_PATH}/infer_python
#bash run.sh <magicmind_model> <batch_size> <image_num>
bash run.sh ${magicmind_model} 1 1000 

```

`infer_python/run.sh`内包含了精度计算模块，关键代码如下：
```
# infer_res_dir由 infer_python/run.sh 创建，
# 例如：mobilenetv3_pytorch_model_force_float32_false_1_infer_res/
# 具体创建规则详见 infer_python/run.sh
function compute_acc(){
    infer_res_dir=${1}
    log_file=${infer_res_dir}/log_eval
    python ${UTILS_PATH}/compute_top1_and_top5.py \
            --result_label_file ${infer_res_dir}/eval_labels.txt \
            --result_1_file ${infer_res_dir}/eval_result_1.txt \
            --result_5_file ${infer_res_dir}/eval_result_5.txt \
            --top1andtop5_file ${infer_res_dir}/eval_result.txt 2>&1 |tee ${log_file}

}

compute_acc  ${infer_res_dir}

```

结果:

```
top1 accuracy: 0.673940
top5 accuracy: 0.872920
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 cd magicmind_cloud/buildin/cv/classification/mobilenetv3_caffe && bash run.sh 来实现一键执行

## 4.高级说明

### 4.1 gen_model 高级说明

PyTorch MobileNet-v3 模型转换为 MagicMind MobileNet-v3 模型分成以下几步：

- 使用 MagicMind Parser 模块将 PyTorch 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

网络特定参数说明:
- `image_dir`: 校准数据文件路径。

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件以及公共参数如`batch_size`, `device_id`的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：
本例使用 MagicMind Python API 编写了名为 infer.py的推理示例程序，其中infer.py将展示如何使用 MagicMind Python API 构建高效的 MobileNet-v3 目标检测(图像预处理=>推理=>图像后处理)。其中程序主要由以下内容构成:

- infer.py: 高效率地将 MagicMind 模型运行在 MLU 板卡上。

参数说明:
- device_id: MLU 设备号
- magicmind_model: MagicMind 模型路径。
- image_dir: 数据集路径
- label_file:ground truth 文件
- result_file:推理结果输出文件 txt 格式
- result_label_file 推理结果输出 label 文件 txt 格式
- result_top1_file:top1 推理结果输出 label 文件 txt 格式
- result_top5_file:top5 推理结果输出 label 文件 txt 格式

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run -h
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

通过精度benchmark测试的脚本跑出 MobileNet-v3 在 imagenet2012 数据集上的50000张图片 top1 和 top5 精度如下
| Model | BatchSize | Precision | top1(%) | top5(%) |
| --------- | ---------- | ---------- | --------- | ---------|
| MobileNet-v3 | 1 | force_float32 | 67.39| 87.29 | 
| MobileNet-v3 | 1 | force_float16 | 67.36 | 87.28 | 
| MobileNet-v3 | 1 | int8_mixed_float16 | 63.69 | 84.98| 

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- mobilenetv3 pth 模型下载链接：https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C

