#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=true
batch_size=16
image_num=1000
infer_mode=infer_python
yolo_mode=val # val or predict

magicmind_model=${MODEL_PATH}/yolov8_onnx_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# 0. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

# 1. gen model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}

# 2 infer
cd $PROJ_ROOT_PATH/infer_python
bash run.sh ${magicmind_model} ${batch_size} ${image_num} ${yolo_mode}
