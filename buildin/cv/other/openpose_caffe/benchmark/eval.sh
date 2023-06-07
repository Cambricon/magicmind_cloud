#!/bin/bash
set -e
set -x

infer_mode=infer_cpp
image_num=0

for backbone in COCO BODY_25;do
# 1. export model
cd ${PROJ_ROOT_PATH}/export_model 
bash run.sh ${backbone}

for precision in qint8_mixed_float16 force_float16 force_float32
do 
    for dynamic_shape in true 
    do 
        for batch_size in 1
        do 
            magicmind_model=${MODEL_PATH}/openpose_caffe_model_${backbone}_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi

            # gen model
            if [ -f ${magicmind_model} ];then
			    rm ${magicmind_model}
            fi
            cd ${PROJ_ROOT_PATH}/gen_model
            bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 

            # infer model and get metric res.
            cd ${PROJ_ROOT_PATH}/${infer_mode}
            bash run.sh  ${magicmind_model} ${batch_size} ${image_num} ${backbone}
        done 
    done 
done
done # backbone
