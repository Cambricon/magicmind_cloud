#!/bin/bash
set -e
set -x

infer_mode=infer_cpp

image_num=50000
# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh

for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape  in true false
    do
        for batch_size in 64
        do
            magicmind_model=${MODEL_PATH}/googlenet_caffe_model_${precision}_${dynamic_shape}
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
 
            cd $PROJ_ROOT_PATH/infer_cpp
            bash run.sh  ${magicmind_model} ${batch_size} ${image_num}
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh  ${magicmind_model} ${batch_size} ${image_num}
        done
    done
done
