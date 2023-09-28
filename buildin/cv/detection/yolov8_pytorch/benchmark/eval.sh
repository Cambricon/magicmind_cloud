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
        for batch_size in 16
        do 
            magicmind_model=${MODEL_PATH}/yolov8_onnx_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi

            # 2. gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${precision} ${batch_size} ${dynamic_shape} ${magicmind_model}
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi

            # 3. infer and calc acc
            cd ${PROJ_ROOT_PATH}/${infer_mode}
            bash run.sh  ${magicmind_model} ${image_num} val 16
        done 
    done 
done




