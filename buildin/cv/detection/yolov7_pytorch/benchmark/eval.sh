#!/bin/bash
set -e
set -x

infer_mode=infer_cpp
image_num=5000

# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

for precision in qint8_mixed_float16 force_float16 force_float32
do 
    for dynamic_shape in true
    do 
        for batch_size in 1
        do 
            magicmind_model=${MODEL_PATH}/yolov7_pytorch_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi
	    infer_res_path="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_${infer_mode}_res"

            # gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi

            # infer and calc acc
            cd ${PROJ_ROOT_PATH}/${infer_mode}
            bash run.sh  ${magicmind_model} ${batch_size} ${image_num} 
        done 
    done 
done




