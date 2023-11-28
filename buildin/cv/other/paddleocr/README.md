# paddleocr

MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将 AI 框架(Tensorflow,PyTorch,ONNX,PaddlePaddle 等) 训练好的算法模型转换成 MagicMind 统一计算图表示,并提供端到端的模型优化、代码生成以及推理业务部署能力。

本 sample 探讨如何将 PaddleOCR 中的 PaddlePaddle 模型实现转换为 MagicMind 模型，进而部署在寒武纪 MLU 板卡上。

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

本例使用的OCR网络模型来自开源项目[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。本项目支持的网络如下：

| 模型  | 配置文件           | 预训练模型 |  
| ----- | ------------------- | ---------- | 
| ch_PP-OCRv3_det_infer | det_mv3_db.yml       | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)          | 
| ch_PP-OCRv3_rec_infer | ch_PP-OCRv3_rec.yml       | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar)          | 
| ch_PP-OCRv2_det_infer | ch_det_res18_db_v2.0.yml | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar)          | 
| ch_PP-OCRv2_rec_infer | rec_chinese_common_train_v2.0.yml | [Model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) |
| ch_ppocr_mobile_v2.0_cls_infer | cls_mv3.yml | [Model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) |

下面将展示如何将该项目中 PaddleOCR 模型转换为 MagicMind 的模型。

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/cv/other/paddleocr
```

在开始运行代码前需要先安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `ICDAR2015_DATASETS_PATH`, 还可以修改 `OCR_VERSION` 来选择模型版本，确认好参数后执行以下命令：

```bash
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

### 3.5 执行推理(会保存可视结果)

1.infer_python

```bash
cd ${PROJ_ROOT_PATH}/infer_python
# bash run.sh <magicmind_model>
bash run_demo.sh ${magicmind_model}
```
**注意：**

infer_python下的脚本区别：

run.sh: 对应模型的精度计算。

run_demo.sh：对应模型的测试图片推理和结果展示demo。

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行 `cd magicmind_cloud/buildin/cv/other/paddleocr && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 gen_model 高级说明

PaddlePaddle 模型转换为 MagicMind 模型分成以下几步：

- 使用paddle2onnx将paddle模型转为onnx格式
- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

参数说明:

**注意：**
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如Onnx）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：[公共组件的README.md](../../../python_common/README.md)

### 4.2 infer_python 高级说明

概述：

本模型推理代码复用了PaddleOCR中的推理代码，并在PaddleOCR加入对magicmind backend的支持，代码见export_model/mm_backend.patch.

本例通过调用PaddleOCR源码中的tools/eval.py来进行模型推理和精度计算，使用tools/infer/predict_det.py/predict_rec.py/predict_cls.py/predict_system.py来进行demo的推理。

参数说明:
- eval.py
- `config`: 模型对应的配置文件路径。
- `Global.use_gpu`: 是否使用gpu，测试中必须设置为false。
- `Global.checkpoints`: 复用为magicmind模型路径。
- `Eval.dataset.data_dir`: 测试数据集路径。
- `Eval.dataset.label_file_list`：测试数据集label路径。

其他参数与PaddleOCR中保持一致。


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

一键运行 benchmark 里的脚本跑出 OCR 网络在 icdar2015 数据集上的精度如下：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash benchmark/eval.sh
```

| Model | Precision           | Batch_Size    | hmean/acc |
| ----- | ------------------- | ----------    | -------- |
| ch_PP-OCRv3_det_infer | force_float32       | 1           | 0.4503 |
| ch_PP-OCRv3_det_infer | force_float16       | 1           | 0.4506 |
| ch_PP-OCRv3_rec_infer | force_float32       | 1           | 0.6076 |
| ch_PP-OCRv3_rec_infer | force_float16       | 1           | 0.6076 |
| ch_PP-OCRv2_det_infer | force_float32       | 1           | 0.4613 |
| ch_PP-OCRv2_det_infer | force_float16       | 1           | 0.4606 |
| ch_PP-OCRv2_rec_infer | force_float32       | 1           | 0.5262 |
| ch_PP-OCRv2_rec_infer | force_float16       | 1           | 0.5257 |

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- icdar2015 数据下载链接： [https://rrc.cvc.uab.es/?ch=4&com=downloads](https://rrc.cvc.uab.es/?ch=4&com=downloads)

- icdar2015 det label下载链接：[https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt)

- icdar2015 rec label下载链接：[https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt](https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt)

- PaddleOCR github 下载链接：[https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
