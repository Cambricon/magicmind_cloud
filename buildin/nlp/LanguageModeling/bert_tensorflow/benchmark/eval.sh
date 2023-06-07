#!/bin/bash
set -e
set -x

for max_seq_length in 384
do 
    cd ${PROJ_ROOT_PATH}/export_model
    bash run.sh
    for precision in force_float32 force_float16
    do 
    for batch in 1
        do
        magicmind_model=${MODEL_PATH}/bert_${precision}_${batch}_false_${max_seq_length}
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch} false ${max_seq_length}
        cd ${PROJ_ROOT_PATH}/infer_python
        bash run.sh ${magicmind_model} ${batch} ${max_seq_length} 
        done
    done
done
