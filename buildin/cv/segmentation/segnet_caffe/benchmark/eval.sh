#!/bin/bash
set -e
set -x

infer_mode=infer_cpp
# set image_num to 0 to use all the images in the dataset.
image_num=0

# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

for precision in qint8_mixed_float16 force_float16 force_float32
do 
    for dynamic_shape in true 
    do 
        for batch_size in 1
        do 
            magicmind_model=${MODEL_PATH}/segnet_caffe_model_${precision}_${dynamic_shape}
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
            bash run.sh  ${magicmind_model} ${batch_size} ${image_num}
        done 
    done 
done

