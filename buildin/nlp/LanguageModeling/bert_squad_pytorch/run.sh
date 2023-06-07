#!/bin/bash
set -e

precision=force_float32
dynamic_shape="false"
batch_size=1
max_seq_length=384
infer_mode=infer_python

magicmind_model=${MODEL_PATH}/bert_squad_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi

# 0. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh $max_seq_length

# 1. gen model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
else
    echo "MagicMind model: ${magicmind_model} already exists!"
fi

if [ ${infer_mode} == "infer_python" ];then
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh  ${magicmind_model} ${batch_size} 
fi

