#!/bin/bash
set -e
set -x

MM_RUN_ENCODER(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_model} \
                          --input_dims ${batch_size},128 ${batch_size} \
                          --devices 0
}

MM_RUN_DECODER(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_model} \
                          --input_dims ${batch_size},80 ${batch_size},1024 ${batch_size},1024 ${batch_size},1024 ${batch_size},1024 ${batch_size},128 ${batch_size},128 ${batch_size},512 ${batch_size},128,512 ${batch_size},128,128 ${batch_size},128 \
                          --devices 0
}

MM_RUN_POSTNET(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_model} \
                          --input_dims ${batch_size},80,512 \
                          --devices 0
}

MM_RUN_WAVEGLOW(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_model} \
                          --input_dims ${batch_size},80,768,1 ${batch_size},8,24576,1 \
                          --devices 0
}

# 1. export model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

for precision in force_float32 force_float16
do 
    for dynamic_shape in true
    do
        for batch_size in 1
        do
            magicmind_encoder_model=${MODEL_PATH}/encoder_${precision}_${dynamic_shape}_model
            magicmind_decoder_model=${MODEL_PATH}/decoder_${precision}_${dynamic_shape}_model
            magicmind_postnet_model=${MODEL_PATH}/postnet_${precision}_${dynamic_shape}_model
            magicmind_waveglow_model=${MODEL_PATH}/waveglow_${precision}_${dynamic_shape}_model
	    if [ ${dynamic_shape} == 'false' ];then
                magicmind_encoder_model="${magicmind_encoder_model}_${batch_size}"
                magicmind_decoder_model="${magicmind_decoder_model}_${batch_size}"
                magicmind_postnet_model="${magicmind_encoder_model}_${batch_size}"
                magicmind_waveglow_model="${magicmind_decoder_model}_${batch_size}"
            fi
            
            # gen model
            if [ ! -f ${magicmind_encoder_model} ] || [ ! -f ${magicmind_decoder_model} ] || [ ! -f ${magicmind_postnet_model} ] || [ ! -f ${magicmind_waveglow_model} ];then  
                cd $PROJ_ROOT_PATH/gen_model
                bash run.sh ${magicmind_encoder_model} ${magicmind_decoder_model} ${magicmind_postnet_model} ${magicmind_waveglow_model} ${precision} ${batch_size} ${dynamic_shape}

            else
                echo "MagicMind model: ${magicmind_encoder_model} ${magicmind_decoder_model} ${magicmind_postnet_model} ${magicmind_waveglow_model} already exists!"
            fi
            
            # run model
            MM_RUN_ENCODER ${magicmind_encoder_model} ${batch_size}
            MM_RUN_DECODER ${magicmind_decoder_model} ${batch_size} 
            MM_RUN_POSTNET ${magicmind_postnet_model} ${batch_size}
            MM_RUN_WAVEGLOW ${magicmind_waveglow_model} ${batch_size}
        done
    done
done
