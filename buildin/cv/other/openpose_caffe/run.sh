#!/bin/bash
set -e
set -x

precision=qint8_mixed_float16
dynamic_shape=false
batch_size=1
image_num=0

# --------- cambricon-note ---------------
# default backbone is COCO, which will have an impact on export_model and infer_cpp.
# if you want to use BODY_25(another valid val of backbone), 
# just uncomment the below line and pass the ${backbone} to export_model/run.sh and infer_cpp/run.sh
#backbone=BODY_25

# for instance:
# cd ${PROJ_ROOT_PATH}/export_model
# bash run.sh ${backbone}
# ...
# cd ${PROJ_ROOT_PATH}/infer_cpp
# bash run.sh  ${magicmind_model} ${batch_size} ${image_num} ${backbone}

#backbone=BODY_25

magicmind_model=${MODEL_PATH}/openpose_caffe_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# 0. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh
#bash run.sh ${backbone}

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
#bash run.sh  ${magicmind_model} ${batch_size} ${image_num} ${backbone}
