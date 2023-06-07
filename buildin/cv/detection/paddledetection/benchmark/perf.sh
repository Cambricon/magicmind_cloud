#!/bin/bash
set -e
set -x
network=paddle_detection

MM_RUN(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run   --magicmind_model ${magicmind_model} \
                            --input_dims ${batch_size},3,640,640 ${batch_size},2 \
                            --devices 0 
}

dynamic_shape=false
# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh


for precision in qint8_mixed_float16 force_float16 force_float32
do 
    for dynamic_shape in false 
    do 
        for batch_size in 1
        do 
            magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi

            # gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi
	    # run model
	    MM_RUN ${magicmind_model} ${batch_size}
        done 
    done 
done


