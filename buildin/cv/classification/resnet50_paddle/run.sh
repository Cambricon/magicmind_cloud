#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=true
batch_size=1
image_num=1000
network=resnet50_paddlecls

magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}
if [ ${dynamic_shape} == 'false' ];then
    magicmind_model="${magicmind_model}_${batch_size}"
fi
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.gen model
if [ ! -f ${magicmind_model} ];then
    cd ${PROJ_ROOT_PATH}/gen_model
    bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape} 
   
else
    echo "MagicMind model: ${magicmind_model} already exists!"
fi

### 2.infer
cd ${PROJ_ROOT_PATH}/infer_python
bash run.sh  ${magicmind_model} ${batch_size} ${image_num}   
