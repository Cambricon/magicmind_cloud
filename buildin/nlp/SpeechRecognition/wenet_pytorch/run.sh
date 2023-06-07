#!/bin/bash
set -e
set -x

precision=force_float32
batch_size=32
dynamic_shape=true
infer_mode=infer_python
magicmind_encoder_model=${MODEL_PATH}/wenet_encoder_pytorch_model_${precision}_${dynamic_shape}
magicmind_decoder_model=${MODEL_PATH}/wenet_decoder_pytorch_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_encoder_model="${magicmind_encoder_model}_${batch_size}"
    magicmind_decoder_model="${magicmind_decoder_model}_${batch_size}"
fi

###0. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###1. gen model
if [ ! -f ${magicmind_encoder_model} ] || [ ! -f ${magicmind_decoder_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${precision} ${batch_size} ${dynamic_shape}
else
    echo "MagicMind model: ${magicmind_encoder_model} and ${magicmind_decoder_model} already exists!"
fi

###2. infer
if [ ${infer_mode} == "infer_python" ] || [ ${dynamic_shape} == 'true' ];then
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${batch_size}
fi
