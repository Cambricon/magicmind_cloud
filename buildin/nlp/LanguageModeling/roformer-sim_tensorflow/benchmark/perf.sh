#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=${1} #force_float32/force_float16
    SHAPE_MUTABLE=${2} #true/false
    BATCH_SIZE=${3}
    MAX_SEQ_LENGTH=${4}
    MM_MODEL=${5}
    if [ ! -d ${PROJ_ROOT_PATH}/data/output ];
    then
        mkdir "${PROJ_ROOT_PATH}/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model ${MM_MODEL} \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --input_files ${PROJ_ROOT_PATH}/data/input0.bin ${PROJ_ROOT_PATH}/data/input1.bin \
                          --devices 0 
}

for max_seq_length in 64
    do
        cd ${PROJ_ROOT_PATH}/export_model
        bash run.sh
        for precision in force_float32 force_float16
        do
            for batch in 2 16 32
            do 
                cd ${PROJ_ROOT_PATH}/gen_model
                MODEL_NAME=${MODEL_PATH}/roformer_${precision}_false_${batch}_${max_seq_length}
                bash run.sh ${MODEL_NAME} ${precision} ${batch} false ${max_seq_length}
                MM_RUN ${precision} false ${batch} ${max_seq_length} ${MODEL_NAME}
            done
        done
done

