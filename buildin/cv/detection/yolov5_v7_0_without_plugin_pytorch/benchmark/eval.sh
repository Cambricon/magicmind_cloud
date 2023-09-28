#!/bin/bash
set -e
set -x

infer_mode=infer_python
image_num=5000

# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

for precision in qint8_mixed_float16 force_float16 force_float32
do 
    for dynamic_shape in true
    do 
        for batch_size in 32
        do 
            magicmind_model=${MODEL_PATH}/yolov5_v7_1_pytorch_model_${precision}_${dynamic_shape}
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

            # infer and calc acc
            cd ${PROJ_ROOT_PATH}/${infer_mode}
            bash run.sh  ${precision} ${dynamic_shape} ${batch_size}
        done 
    done 
done




