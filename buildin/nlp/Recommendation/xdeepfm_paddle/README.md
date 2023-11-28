# xdeepfm_paddle

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX,PaddlePaddle 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 Paddle_xDeepFM 中的 PaddlePaddle 模型实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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
- [高级说明](#4高级说明)
  - [gen_model 高级说明](#41-gen_model-高级说明)
  - [infer_python 高级说明](#42-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本例使用的 xDeepFM 网络模型来自开源项目 [PaddleRec](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/xdeepfm)。
下面将展示如何将该项目中 Paddle_xDeepFM 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/nlp/Recommendation/xdeepfm_paddle
```

在开始运行代码前需要先安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -i https://mirror.baidu.com/pypi/simple -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：   

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
#用户需要提前准备好 xDeepFM dygraph model
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd ${PROJ_ROOT_PATH}/gen_model
# bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ${MODEL_PATH}/xdeepfm_paddle_model_force_float32_true force_float32 1 true
```

### 3.5 执行推理

```bash
cd ${PROJ_ROOT_PATH}/infer_python
# bash run.sh <magicmind_model> <batch_size> <input_fetature_num>
bash run.sh ${MODEL_PATH}/xdeepfm_paddle_model_force_float32_true 512 800000
```
结果：   
```
auc: 0.791781
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/nlp/Recommendation/xdeepfm_paddle && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

Paddle_xDeepFM 模型转换为 MagicMind 模型分成以下几步：

- 使用 paddle2onnx 将 paddle 模型转为onnx格式
- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如Onnx）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本例使用 MagicMind PYTHON API 编写了名为 infer_python 的 CTR（点击率）预测程序。   

参数说明:
- device_id: 设备号。
- magicmind_model: MagicMind 模型路径。
- dataset_dir: 输入特征的目录。
- sample_num: 输入样本的数量。
- batch_size: 生成可变模型时 batch_size 可以在 dimension range 内取值，生成不可变模型时 batch_size 的取值需要对应 onnx 的输入维度。


## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本仓库通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model ${MM_MODEL} --batch_size ${BATCH_SIZE} --devices ${DEV_ID} --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

### 5.2 精度 benchmark 测试

一键运行 benchmark 里的脚本跑出 xDeepFM 网络在 criteo 全量数据集上的精度如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

| Model | Precision           | Batch_Size    | AUC |
| ----- | ------------------- | ----------    | -------- |
| xDeepFM | force_float32       | 512           | 0.793927 |
| xDeepFM | force_float16       | 512           | 0.793914 |
| xDeepFM | qint8_mixed_float16 | 512           | 0.791854 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- Criteo 数据集下载链接： [https://www.kaggle.com/c/criteo-display-ad-challenge/](https://www.kaggle.com/c/criteo-display-ad-challenge/)

- PaddleRec github 下载链接：[https://github.com/PaddlePaddle/PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
