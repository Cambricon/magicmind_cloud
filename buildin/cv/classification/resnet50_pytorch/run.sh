#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape="false"
batch_size=1
image_num=1000
infer_mode=infer_python

cd $PROJ_ROOT_PATH/export_model
bash run.sh

magicmind_model=${MODEL_PATH}/resnet50_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# gen model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
fi

# infer 
if [ ${infer_mode} == "infer_python" ];then
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${magicmind_model} ${batch_size} ${image_num}  
fi