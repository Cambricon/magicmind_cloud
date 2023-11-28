# 基于 MagicMind 适配部署 WeNet 语音识别


Wenet 是出门问问语音团队联合西工大语音实验室开源的一款面向工业落地应用的语音识别工具包，该工具用一套简洁的方案提供了语音识别从训练到部署的一条龙服务。
MagicMind 是寒武纪的推理引擎，用于在寒武纪硬件产品上加速 AI 推理任务。下面我们使用 MagicMind 来实现基于 WeNet 工具箱中 Conformer  模型的高性能语音识别任务。


## 目录

- [模型概述](#1-模型概述)
- [前提条件](#2-前提条件)
- [快速使用](#3-快速使用)
  - [环境准备](#31-环境准备)
  - [下载仓库](#32-下载仓库)
  - [数据集、模型准备 & 模型转换](#33-数据集模型准备-模型转换)
  - [编译 MagicMind 模型](#34-编译-magicmind-模型)
  - [执行推理](#35-执行推理)
  - [一键运行](#36-一键运行)
- [高级说明](#4-高级说明)
  - [export_model 高级说明](#41-export_model高级说明)
  - [gen_model 高级说明](#42-gen_model-高级说明)
  - [infer_python 高级说明](#43-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [精度 benchmark 结果](#51-精度-benchmark-结果)
  - [性能 benchmark 结果](#52-性能-benchmark-结果)
- [免责声明](#6-免责声明)

## 1. 模型概述
Conformer 是 Google 在 2020 年提出的语音识别模型，主要结合了 CNN 和 Transformer 的优点，其中 CNN 能高效获取局部特征，而 Transformer 在提取长序列依赖的时候更有效。
Conformer 则是将卷积应用于 Transformer 的 Encoder 层，用卷积加强 Transformer 在 ASR 领域的效果。
论文链接：[Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100.pdf)。

本示例中的 WeNet/conformer 模型基于 [WeNet 官方权重](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz) 

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/nlp/SpeechRecognition/WeNet_pytorch
```

在开始运行代码前需要执行以下命令安装必要的库：

```baah
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量，根据数据集实际路径修改 `env.sh` 内的 `AISHELL_DATASETS_PATH`, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 数据集、模型准备 & 模型转换

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```


### 3.4 编译 MagicMind 模型

```bash
magicmind_encoder_model=${MODEL_PATH}/wenet_encoder_pytorch_model_${precision}_${dynamic_shape}
magicmind_decoder_model=${MODEL_PATH}/wenet_decoder_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_encoder_model="${magicmind_encoder_model}_${batch_size}"
    magicmind_decoder_model="${magicmind_decoder_model}_${batch_size}"
fi
cd $PROJ_ROOT_PATH/gen_model
# bash run.sh <magicmind_encoder_model> <magicmind_decoder_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} force_float32 32 true
```

### 3.5 执行推理

1. infer.py
```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <magicmind_encoder_model> <magicmind_decoder_model>
bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model}
```
2. 计算精度

```bash
python ${PROJ_ROOT_PATH}/export_model/wenet/tools/compute-wer.py --char=1 --v=0 ${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0/data/test/text  ${PROJ_ROOT_PATH}/data/output/infer_python_output_force_float32  2>&1 | tee $PROJ_ROOT_PATH/data/output/force_float32_log_eval
```

WER 结果：

```bash
Overall -> 4.68 % N=104765 C=100092 S=4553 D=120 I=225
```

### 3.6 一键运行

以上 3.3~3.6 的步骤也可以通过运行 `cd magicmind_cloud/buildin/nlp/SpeechRecognition/WeNet_pytorch && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 export_model 高级说明

本例使用 PyTorch 训练好的模型，通过将 PyTorch 模型转换为 ONNX 模型再转换为 MagicMind 模型进行部署。
首先将以下 WeNet 官方预训练模型文件下载至`$MODEL_PATH`目录：

```bash
cd $MODEL_PATH
wget http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz
tar zxvf 20211025_conformer_exp.tar.gz
```

然后下载 WeNet 源码将 pytorch 模型转换成 onnx 模型:

```bash
cd $PROJ_ROOT_PATH/export_model
if [ -x wenet ]; then
    echo "WeNet Official repo already exists."
else
    echo "get WeNet..."
    git clone -b v2.0.0 https://github.com/wenet-e2e/wenet.git
    git apply  patchs/mlu.patch
fi

### pytorch models convert to onnx models
cd $PROJ_ROOT_PATH/export_model/wenet
python ./wenet/bin/export_onnx_gpu.py --config ${MODEL_PATH}/20211025_conformer_exp/train.yaml \
                                   --checkpoint ${MODEL_PATH}/20211025_conformer_exp/final.pt  \
                                   --beam 4  \
                                   --output_onnx_dir ${MODEL_PATH}/20211025_conformer_exp/onnx_model/ \
                                   --cmvn_file ${MODEL_PATH}/20211025_conformer_exp/global_cmvn
```

### 4.2 gen_model 高级说明
ONNX WeNet 模型转换为 MagicMind WeNet 模型分成以下几步：

- 使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
- 模型量化。
- 使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

注意：
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：公共组件的README.md

### 4.3 infer_python 高级说明

参数说明：
- `device_id`: 设备号。
- `config`: 模型配置文件。
- `test_data`: aishell 数据集列表文件。
- `dict`: 字典文件。
- `mode`: 推理模式, 如 attention_rescoring 。
- `precision`: 精度模式，当前只支持 force_float32。
- `encoder_magicmind`: encoder MagicMind 模型。
- `decoder_magicmind`: decoder MagicMind 模型。
- `result_file`: 推理结果保存文件。

## 5. 精度和性能 benchmark

### 5.1 精度 benchmark 结果

|model|dataset|test set|decoding method|WER|
|---|---|---|---|---|
|conformer|aishell|test|attention rescoring|4.68|


本 sample 通过一键运行 benchmark 里的脚本得到精度 benchmark 结果：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash eval.sh
```

### 5.2 性能 benchmark 结果

本 sample 通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据。可变模型需要用户指定input_dims或batch_size。

```bash
#查看参数说明
mm_run --h
mm_run --magicmind_model $MM_MODEL --devices $DEV_ID --threads 1 --iterations 1000
```

或者通过一键运行 benchmark 里的脚本：

```bash
cd ${PROJ_ROOT_PATH}/benchmark
bash perf.sh
```

## 6 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* Wenet 官方仓库链接：https://github.com/wenet-e2e/wenet 

* conformer 权重链接：http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz

