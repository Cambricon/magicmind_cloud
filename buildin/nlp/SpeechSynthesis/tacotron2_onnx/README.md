# 基于 Magicmind 适配部署 Tacotron2+Waveglow 语音合成

Tacotron2 和 WaveGlow 模型构成了一个文本语音(TTS)系统，使用户能够合成自然语音。
MagicMind 是寒武纪的推理引擎，用于在寒武纪硬件产品上加速 AI 推理任务。下面我们使用 MagicMind 来实现基于 Tacotron2 和 WaveGlow 模型的高性能语音合成任务。

本 demo 探讨如何使用 MagicMind 来在寒武纪 MLU370 板卡上适配和部署 Tacotron2+Waveglow 模型。

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
  - [infer_python 高级说明](#43-infer_python-高级说明)
- [精度和性能 benchmark](#5-精度和性能-benchmark)
  - [性能 benchmark 测试](#51-性能-benchmark-测试)
  - [精度 benchmark 测试](#52-精度-benchmark-测试)
- [免责声明](#6-免责声明)

## 1. 模型概述

本示例中的 tacotron2 模型基于 [tacotron2 checkpoint](https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427) 和 [waveglow checkpoint](https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/19.10.0/files/nvidia_waveglow256pyt_fp16)

## 2. 前提条件

请移至[主页面 README.md](../../../../README.md)的`2.前提条件`

## 3. 快速使用

### 3.1 环境准备

请移至[主页面 README.md](../../../../README.md)的`3.环境准备`

### 3.2 下载仓库

```bash
# 下载仓库
git clone 本仓库
cd magicmind_cloud/buildin/nlp/SpeechSynthesis/tacotron2_onnx
```

在安装依赖前需要先安装下载源：

```bash
pip install nvidia-pyindex
```

在开始运行代码前需要先安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ -f https://download.pytorch.org/whl/torch_stable.html
```

在开始运行代码前需要先检查 `env.sh` 里的环境变量, 并且执行以下命令：

```bash
source env.sh
```

### 3.3 准备数据集和模型

```bash
cd $PROJ_ROOT_PATH/export_model
bash run.sh
```

### 3.4 编译 MagicMind 模型

生成 float16 精度模型并指定 batchsize 范围: 指定 batchsize 可运行范围为 1 到 16，并且优化 batchsize=4 的规模。

```bash
magicmind_encoder_model=${MODEL_PATH}/tacotron_encoder_pytorch_model_${precision}_${dynamic_shape}
magicmind_decoder_model=${MODEL_PATH}/tacotron_decoder_pytorch_model_${precision}_${dynamic_shape}
magicmind_postnet_model=${MODEL_PATH}/tacotron_postnet_pytorch_model_${precision}_${dynamic_shape}
magicmind_waveglow_model=${MODEL_PATH}/tacotron_waveglow_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_encoder_model="${magicmind_encoder_model}_${batch_size}_${seq_len}"
    magicmind_decoder_model="${magicmind_decoder_model}_${batch_size}_${seq_len}"
    magicmind_postnet_model="${magicmind_postnet_model}_${batch_size}_${seq_len}"
    magicmind_waveglow_model="${magicmind_waveglow_model}_${batch_size}_${seq_len}"
fi
cd $PROJ_ROOT_PATH/gen_model
# bash run.sh <magicmind_encoder_model> <magicmind_decoder_model> <magicmind_postnet_model> <magicmind_waveglow_model> <precision> <batch_size> <dynamic_shape>
bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${magicmind_postnet_model} ${magicmind_waveglow_model} force_float32 4 true
```

### 3.5 执行推理

```bash
cd $PROJ_ROOT_PATH/infer_python
# bash run.sh <magicmind_encoder_model> <magicmind_decoder_model> <magicmind_postnet_model> <magicmind_waveglow_model> <precision> <batch_size>
bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${magicmind_postnet_model} ${magicmind_waveglow_model} force_float32 4
```

### 3.6 一键运行

以上 3.3~3.5 的步骤也可以通过运行` cd magicmind_cloud/buildin/nlp/SpeechSynthesis/tacotron2_onnx && bash run.sh` 来实现一键执行

## 4. 高级说明

### 4.1 export_model 高级说明

本例使用 PyTorch 训练好的模型，通过将 PyTorch 模型转换为 ONNX 模型再转换为 MagicMind 模型进行部署。
首先将以下 tacotron2 和 waveglow 模型文件下载至`$MODEL_PATH`目录：

```bash
cd $MODEL_PATH
wget -c https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427
wget -c https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/19.10.0/files/nvidia_waveglow256pyt_fp16
```

然后下载 tensorrt 源码将 pytorch 模型转换成 onnx 模型:

```bash
# install tensorrt for convert torch to ONNX IRs
cd $PROJ_ROOT_PATH/export_model
if [ -f tensorrt.tar.gz ]; then
    echo "tensorrt.tar.gz already exists."
else
    echo "Downloading TensorRT"
    wget -c https://github.com/NVIDIA/TensorRT/archive/refs/tags/22.03.tar.gz -O tensorrt.tar.gz
fi

pushd /tmp
    cp $PROJ_ROOT_PATH/export_model/tensorrt.tar.gz .
    tar -zxf tensorrt.tar.gz
    if [ ! -d TensorRT ];
    then
        mv TensorRT-22.03 TensorRT
        patch /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_tacotron22onnx.py $PROJ_ROOT_PATH/export_model/patchs/convert_tacotron22onnx.diff
        patch /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_waveglow2onnx.py $PROJ_ROOT_PATH/export_model/patchs/convert_waveglow22onnx.diff
    fi
popd

apt update
if [ $? -ne 0 ]; then
    echo 'apt update failed, check your network connection please!!!'
    exit 1
fi
apt-get install libsndfile1 -y
if [ $? -ne 0 ]; then
    echo 'apt-get install libsndfile1 failed, check your network connection please!!!'
    exit 1
fi

### pytorch models convert to onnx models
# tacotron2
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_tacotron22onnx.py --tacotron2 $MODEL_PATH/nvidia_tacotron2pyt_fp16_20190427 -o $MODEL_PATH
echo "tacotron2 checkpoint converts to onnx models"
# waveglow
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_waveglow2onnx.py --waveglow $MODEL_PATH/nvidia_waveglow256pyt_fp16 --config-file /tmp/TensorRT/demo/Tacotron2/config.json --wn-channels 256 -o $MODEL_PATH
echo "waveglow checkpoint converts to onnx models"
```

### 4.2 gen_model 高级说明
ONNX Tacotron 模型转换为 MagicMind Tacotron 模型分成以下几步：

使用 MagicMind Parser 模块将 onnx 文件解析为 MagicMind 网络结构。
模型量化。
使用 MagicMind Builder 模块生成 MagicMind 模型实例并保存为离线模型文件。

注意：
在gen_model内使用了一些公共的组件，例如arg解析、第三方框架（如PyTorch）模型解析、MagicMind 配置设定等，这些公共组件的说明详见：公共组件的README.md


### 4.3 infer_python 高级说明

参数说明：

- `devices_id`: 指定使用的 MLU 设备 id
- `encoder_magicmind`: encoder magicmind 模型文件目录。
- `decoder_magicmind`: decoder magicmind 模型文件目录。
- `postnet_magicmind`: postnet magicmind 模型文件目录。
- `waveglow_magicmind`: waveglow_magicmind 模型文件目录。
- `batch_size`: batch_size
- `il`: input length, 指定测试的输入文本长度。
- `precision`: 精度模式，如 force_float32，force_float16。
- `no-waveglow`: 不运行 waveglow 网络片段。

## 5. 精度和性能 benchmark

### 5.1 性能 benchmark 测试

本 sample 通过寒武纪提供的 Magicmind 性能测试工具 mm_run 展示性能数据。可变模型需要用户指定input_dims或batch_size。

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

暂无

## 6. 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

- checkpoint:nvidia_tacotron2pyt_fp16_20190427 下载链接: [https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427](https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427)
- checkpoint:nvidia_waveglow256pyt_fp16 下载链接：[https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/19.10.0/files/nvidia_waveglow256pyt_fp16](https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/19.10.0/files/nvidia_waveglow256pyt_fp16)
- tensorrt 实现源码下载链接：[https://github.com/NVIDIA/TensorRT/archive/refs/tags/22.03.tar.gz](https://github.com/NVIDIA/TensorRT/archive/refs/tags/22.03.tar.gz)
