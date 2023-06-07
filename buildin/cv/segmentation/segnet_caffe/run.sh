#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=false
batch_size=16
# cambricon-note: set image_num to 0 to use all the images in the dataset
image_num=0

magicmind_model=${MODEL_PATH}/segnet_caffe_model_${precision}_${dynamic_shape}
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

# 2 infer cpp
cd ${PROJ_ROOT_PATH}/infer_cpp
bash run.sh  ${magicmind_model} ${batch_size} ${image_num}
