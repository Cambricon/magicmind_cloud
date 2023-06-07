#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=false
batch_size=1
network=paddle_detection

### 0. export model
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
for precision in force_float32 
do
    magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}
    if [ ${dynamic_shape} == 'false' ];then
        magicmind_model="${magicmind_model}_${batch_size}"
    fi
    ### 1.gen model
    if [ ! -f ${magicmind_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
    else
        echo "MagicMind model: ${magicmind_model} already exists!"
    fi
    ### 2 infer_python(include eval)
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${magicmind_model}
done