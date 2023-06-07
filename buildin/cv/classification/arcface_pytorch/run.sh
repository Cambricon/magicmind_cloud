#!/bin/bash
set -e
set -x

precision=qint8_mixed_float16
dynamic_shape=false
batch_size=1
image_num=1000

magicmind_model=${MODEL_PATH}/arcface_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

###1. convert torch.jit.trace model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. build magicmind model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
   
else
    echo "MagicMind model: ${magicmind_model} already exists!"
fi
###3. infer
cd ${PROJ_ROOT_PATH}/infer_cpp
bash run.sh  ${magicmind_model} ${batch_size} ${image_num}  
