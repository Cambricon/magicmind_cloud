#!/bin/bash
set -e
set -x

MM_RUN(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run   --magicmind_model ${magicmind_model} \
                            --batch_size ${batch_size} \
                            --devices 0 
}

dynamic_shape=false
# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh 1
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape in false 
    do
        for batch_size in 1 16 32
        do 
            magicmind_model=${MODEL_PATH}/retinaface_pytorch_model_${precision}_${dynamic_shape}_${batch_size}
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}    
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi
            MM_RUN ${magicmind_model} ${batch_size}
        done
    done
done


