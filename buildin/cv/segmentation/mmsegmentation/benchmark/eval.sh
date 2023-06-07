#!/bin/bash
set -e
set -x

# 0.export onnx model
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh

# unet不支持bs>=8
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape in false
    do
        for batch_size in 4
        do
            magicmind_model=${MODEL_PATH}/${MMSEGMENTATION_MODEL_NAME}_mmsegmentation_model_${precision}_${dynamic_shape}
            if [ ${dynamic_shape} == 'false' ];then
                magicmind_model="${magicmind_model}_${batch_size}"
            fi

            # 1.gen model
            if [ ! -f ${magicmind_model} ];then
                cd ${PROJ_ROOT_PATH}/gen_model
                bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}    
            else
                echo "MagicMind model: ${magicmind_model} already exists!"
            fi

            # 2.infer python
            cd ${PROJ_ROOT_PATH}/infer_python
            bash run.sh  ${magicmind_model} ${batch_size}
        done
    done
done
