#!/bin/bash
set -e
set -x

### download datasets and models
bash $PROJ_ROOT_PATH/export_model/get_datasets_and_models.sh

### install dependencies
# install tensorrt for convert torch to ONNX IRs
cd $PROJ_ROOT_PATH/export_model
if [ -f tensorrt.tar.gz ]; then
    echo "tensorrt.tar.gz already exists."
    ls /tmp
    #mkdir /tmp
    cp tensorrt.tar.gz /tmp
else
    echo "here"
    wget -c https://github.com/NVIDIA/TensorRT/archive/refs/tags/22.03.tar.gz -O tensorrt.tar.gz
fi

pushd /tmp
    tar -zxf tensorrt.tar.gz
    mv TensorRT-22.03 TensorRT
    patch /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_tacotron22onnx.py $PROJ_ROOT_PATH/export_model/patchs/convert_tacotron22onnx.diff
    patch /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_waveglow2onnx.py $PROJ_ROOT_PATH/export_model/patchs/convert_waveglow22onnx.diff
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
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_tacotron22onnx.py --tacotron2 $PROJ_ROOT_PATH/data/models/nvidia_tacotron2pyt_fp16_20190427 -o $PROJ_ROOT_PATH/data/models/
echo "tacotron2 checkpoint converts to onnx models"
# waveglow
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_waveglow2onnx.py --waveglow $PROJ_ROOT_PATH/data/models/nvidia_waveglow256pyt_fp16 --config-file /tmp/TensorRT/demo/Tacotron2/config.json --wn-channels 256 -o $PROJ_ROOT_PATH/data/models/
echo "waveglow checkpoint converts to onnx models"
