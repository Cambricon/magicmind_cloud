#!/bin/bash
set -e
set -x

network=dbnet_pytorch
dynamic_shape=true
batch_size=1

cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}    
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
    if [ ! -f ${magicmind_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
    else
        echo "MagicMind model: ${magicmind_model} already exists!"
    fi
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${magicmind_model}
done
