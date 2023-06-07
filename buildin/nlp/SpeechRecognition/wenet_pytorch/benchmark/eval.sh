#!/bin/bash
set -e
set -x

infer_mode=infer_python

# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi


for precision in force_float32
do
    for dynamic_shape in true
    do
        for batch_size in 32
        do
            magicmind_encoder_model=${MODEL_PATH}/wenet_encoder_pytorch_model_${precision}_${dynamic_shape}
            magicmind_decoder_model=${MODEL_PATH}/wenet_decoder_pytorch_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_encoder_model="${magicmind_encoder_model}_${batch_size}"
                magicmind_decoder_model="${magicmind_decoder_model}_${batch_size}"
            fi
 
            # gen model
            if [ ! -f ${magicmind_encoder_model} ] || [ ! -f ${magicmind_decoder_model} ];then
                cd $PROJ_ROOT_PATH/gen_model
                bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${precision} ${batch_size} ${dynamic_shape}
            else
                echo "MagicMind model: ${magicmind_encoder_model} and ${magicmind_decoder_model} already exists!"
                fi

            # infer and calc acc
            cd ${PROJ_ROOT_PATH}/${infer_mode}
            bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${batch_size} 
            
        done
    done
done

