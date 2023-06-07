#!/bin/bash
set -e
set -x

precision=force_float32
dynamic_shape=true
batch_size=1
network=paddle_ocr

### 0. export model
cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
for precision in force_float32 force_float16 
do
    magicmind_det_model=${MODEL_PATH}/${network}_det_model_${precision}_${dynamic_shape}
    if [ ${dynamic_shape} == 'false' ];then
        magicmind_det_model="${magicmind_det_model}_${batch_size}"
    fi
    magicmind_rec_model=${MODEL_PATH}/${network}_rec_model_${precision}_${dynamic_shape}
    if [ ${dynamic_shape} == 'false' ];then
        magicmind_rec_model="${magicmind_rec_model}_${batch_size}"
    fi
    magicmind_cls_model=${MODEL_PATH}/${network}_cls_model_${precision}_${dynamic_shape}
    if [ ${dynamic_shape} == 'false' ];then
        magicmind_cls_model="${magicmind_cls_model}_${batch_size}"
    fi
    ### 1.gen model
    if [ ! -f ${magicmind_det_model} ] || [ ! -f ${magicmind_rec_model} ] || [ ! -f ${magicmind_cls_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_det_model} ${magicmind_rec_model} ${magicmind_cls_model} ${precision} ${batch_size} ${dynamic_shape}
    else
        echo "MagicMind model: ${magicmind_det_model} , ${magicmind_rec_model} and ${magicmind_cls_model} already exists!"
    fi
    ### 2 infer_python(include eval)
    cd ${PROJ_ROOT_PATH}/infer_python
    bash run.sh ${magicmind_det_model} ${magicmind_rec_model} ${magicmind_cls_model}
done