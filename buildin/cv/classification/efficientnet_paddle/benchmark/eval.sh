#!/bin/bash
set -e
set -x

image_num=50000
dynamic_shape=true
batch_size=32
network=efficientnet_paddlecls

cd $PROJ_ROOT_PATH/export_model
bash run.sh
#dynamic
for precision in force_float32 force_float16 qint8_mixed_float16
do    
    magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}
    if [ ${dynamic_shape} == 'false' ];then
        magicmind_model=${magicmind_model}_${batch_size}
    fi
    if [ ! -f ${magicmind_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
    else
        echo "MagicMind model: ${magicmind_model} already exists!"
    fi	

    cd $PROJ_ROOT_PATH/infer_python
    bash run.sh  ${magicmind_model} ${batch_size} ${image_num} 
    
done