#!/bin/bash
set -e
set -x

# 0.export onnx model
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh

img_num=1000

# 部分模型可能不支持dynamic_shape为true或者batch_size>1
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape in false
    do
        for batch_size in 1 8
        do
            magicmind_model=${MODEL_PATH}/${MMDETECTION_MODEL_NAME}_mmdetection_model_${precision}_${dynamic_shape}
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
            bash run.sh  ${magicmind_model} ${batch_size} ${img_num}
        done
    done
done
