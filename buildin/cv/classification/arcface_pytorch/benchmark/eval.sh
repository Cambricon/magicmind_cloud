#!/bin/bash
set -e
set -x

batch_size=128
dynamic_shape=false
network=arcface_pytorch

cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
cd ${PROJ_ROOT_PATH}/infer_cpp
bash build.sh

for precision in force_float32 force_float16 qint8_mixed_float16
do
    magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}_${batch_size}  
    if [ ! -f ${magicmind_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
    else
        echo "MagicMind model: ${magicmind_model} already exists!"
    fi
    cd ${PROJ_ROOT_PATH}/infer_cpp
    bash run.sh ${magicmind_model} ${batch_size} 469375
    output_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_acc"   
    python ${UTILS_PATH}/ijbc_eval.py --features_dir ${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res \
                                  --output_file ${output_dir} \
                                  --face_tid_mid_file ${IJB_DATASETS_PATH}/IJBC/meta/ijbc_face_tid_mid.txt \
                                  --template_pair_label_file ${IJB_DATASETS_PATH}/IJBC/meta/ijbc_template_pair_label.txt
done
