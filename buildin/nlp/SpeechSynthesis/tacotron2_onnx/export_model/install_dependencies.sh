#!/bin/bash

# The onnx-graphsurgeon package in the PyPI repository is merely a placeholder project
# while the actual package is located in the NVIDIA Python Package Index.
pip install nvidia-pyindex -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install onnx-graphsurgeon -i https://pypi.tuna.tsinghua.edu.cn/simple/
# install tensorrt for convert torch to ONNX IRs
cd $PROJ_ROOT_PATH/export_model
if [ -f tensorrt.tar.gz ]; then
    echo "tensorrt.tar.gz already exists."
else
    echo "Downloading TensorRT.."
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
