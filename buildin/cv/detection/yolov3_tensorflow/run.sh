#!/bin/bash
set -e
set -x

#precision=force_float16
#precision=force_float32
precision=qint8_mixed_float16
dynamic_shape=false
batch_size=16
image_num=5000
#image_num=1

magicmind_model=${MODEL_PATH}/yolov3_tensorflow_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# 0. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

# 1. gen model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
   
else
    echo "MagicMind model: ${magicmind_model} already exists!"
fi

# 2 infer 
cd ${PROJ_ROOT_PATH}/infer_cpp
bash run.sh  ${magicmind_model} ${batch_size} ${image_num}  
