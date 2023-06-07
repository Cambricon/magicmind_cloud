#!/bin/bash
set -e
set -x

precision_mode=$1
onnx_path=${MODEL_PATH}/ocrnet.onnx
output_model=${MODEL_PATH}/ocrnet_${precision_mode}.mm
quanti_image_dir=$PROJ_ROOT_PATH/export_model/mmsegmentation/demo
# 检查是否生成了ocrnet.onnx 文件
if [ ! -f ${onnx_path} ];then
    echo 'Please export onnx model first. in export_model dir, ./run.sh'
    exit 1
fi

if [ ! -f ${output_model} ];then
    python gen_model.py --onnx_model ${onnx_path} \
                        --precision ${precision_mode} \
                        --image_dir ${quanti_image_dir} \
                        --output_model ${output_model}
    echo "Generate magicmind model finish, save to "${output_model}
else
    echo ${output_model}" already exist."
fi