# Tensorflow-Roformer-sim 通过 MagicMind 适配和部署

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何使用 MagicMind 来在寒武纪 MLU370 板卡上适配和部署 Roformer-sim 模型。

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
  - [gen_model 高级说明](#41gen_model-高级说明)
  - [infer_python 高级说明](#42infer_python-高级说明)
- [精度和性能 benchmark](#5精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
- [免责声明](#6.免责声明)

## 1.模型概述

本示例中的 Roformer-sim 模型来自于 [https://github.com/ZhuiyiTechnology/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)。下面将展示如何将该项目中的Tensorflow实现的roformer-sim模型转换为Magicmind的模型。

其中 MAX_SEQ_LENGTH 为 64。

## 2.前提条件

请移至[主页面](../../../../README.md)的`2.前提条件`。

## 3.快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd buildin/nlp/LanguageModeling/roformer-sim_tensorflow
```


在开始运行代码前需要先检查 env.sh 里的环境变量，并且执行以下命令：

```bash
source env.sh
pip uninstall keras-nightly
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

注意：在安装requirement之前需要先卸载keras-nightly。


### 3.3 准备数据集和模型

- 准备模型

将原生的 tensorflow checkpoint 转换为本仓库所需要的，其中MAX_SEQ_LENGTH 为 64，batch_size实际推理时输入设置为2。

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

```bash
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape> <max_seq_length>
bash run.sh ${MODEL_PATH}/roformer_force_float32_false_2_8 force_float32 2 false 8
```
结果：

```bash
Generate model done, model save to magicmind_cloud/buildin/nlp/LanguageModeling/roformer-sim_tensorflow/data/models/roformer-sim_tf_force_float32_false_2_64
```
### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <model_path>
bash run.sh ${MODEL_PATH}/roformer_force_float32_false_2_8
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行`cd magicmind_cloud/buildin/nlp/LanguageModeling/roformer-sim_tensorflow && bash run.sh` 来实现一键执行

## 4.高级说明

### 4.1gen_model 高级说明

Tensorflow roformer-sim 模型转换为 MagicMind roformer-sim 模型分成以下几步：

- 使用 MagicMind Parser 模块将 pb 文件解析为 MagicMind 网络结构。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

- `pb_model`: Tensorflow pb 模型的路径。
- `output_model`: 保存 MagicMind 模型路径。
- `precision`: 精度模式，如 force_float32，force_float16。
- `shape_mutable`: 是否生成可变 batch_size 的 MagicMind 模型。
- `batch_size`: 生成可变模型时 batch_size 可以在dim_range范围内随意取值，生成不可变模型时 batch_size 的取值需要对应 pb 的输入维度。
- `max_seq_length`: max_seq_length。

### 4.2infer_python 高级说明

本例使用 MagicMind PYTHON API 编写了名为 infer_python 的测试程序。infer_python 将展示如何使用 MagicMind PYTHON API 构建高效的 roformer-sim demo(预处理=>推理=>后处理)。
MagicMind 提供推理能力的类为 Engine 和 Context。其中一个 Engine 实例可使用一张 MLU 板卡。一个 Engine 实例可创建多个 Context 实例来向 MLU 下发任务。一个 Context 实例可使用一个 Queue 实例下发任务，同一个 Context 不能通过多个不同的 Queue 下发任务。多个 Context 实例可以通过同一个 Queue 实例下发任务。更多详细信息请参考《Cambricon-MagicMind-User-Guide》中编程模型章节。

参数说明：

- `device_id`: 设备号。
- `magicmind_model`: MagicMind 模型路径。
- `vocab_file`: 字典文件。

## 5.精度和性能 benchmark

### 5.1 性能 benchmark 测试

本 sample 通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据。可变模型需要用户指定input_dims或batch_size。


```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --batch_size $BATCH_SIZE  --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

## 6.免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- roformer-sim github：https://github.com/ZhuiyiTechnology/roformer-sim
- 模型下载链接：https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip
