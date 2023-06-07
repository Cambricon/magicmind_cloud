#!/bin/bash
set -e
set -x

### install dependencies
bash $PROJ_ROOT_PATH/export_model/install_dependencies.sh

### download datasets and models
bash $PROJ_ROOT_PATH/export_model/get_datasets_and_models.sh

### pytorch models convert to onnx models
# tacotron2
if [ ! -d $MODEL_PATH ];
then
    mkdir -p $MODEL_PATH
fi 
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_tacotron22onnx.py --tacotron2 $MODEL_PATH/nvidia_tacotron2pyt_fp16_20190427 -o $MODEL_PATH
echo "tacotron2 checkpoint converts to onnx models"
# waveglow
python /tmp/TensorRT/demo/Tacotron2/tensorrt/convert_waveglow2onnx.py --waveglow $MODEL_PATH/nvidia_waveglow256pyt_fp16 --config-file /tmp/TensorRT/demo/Tacotron2/config.json --wn-channels 256 -o $MODEL_PATH
echo "waveglow checkpoint converts to onnx models"
