#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=false
batch_size=1
image_num=-1
infer_mode=infer_cpp
#infer_mode=infer_python

magicmind_model=${MODEL_PATH}/yolov7_pytorch_model_${precision}_${dynamic_shape}
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
if [ ${infer_mode} == "infer_python" ];then
    cd ${PROJ_ROOT_PATH}/infer_python
    echo "cambricon-note: infer python"
    bash run.sh  ${magicmind_model} ${batch_size} ${image_num}
elif [ ${infer_mode} == "infer_cpp" ];then
    cd ${PROJ_ROOT_PATH}/infer_cpp
    echo "cambricon-note: infer cpp"
    bash run.sh  ${magicmind_model} ${batch_size} ${image_num}  
fi
