#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1 #force_float32/force_float16
    SHAPE_MUTABLE=$2 #true/false
    BATCH_SIZE=$3
    MAX_SEQ_LENGTH=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/bert_tensorflow_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --devices 0 \
                          --input_files $PROJ_ROOT_PATH/data/input_0_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input_1_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input_2_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin 
}

for max_seq_length in 384
do
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for precision in force_float32 force_float16
    do
        for batch in 1 16 32 
        do 
            cd ${PROJ_ROOT_PATH}/gen_model
            magicmind_model=${MODEL_PATH}/bert_${precision}_${batch}_false_${max_seq_length}
            bash run.sh ${magicmind_model} ${precision} ${batch} false ${max_seq_length}
            MM_RUN $precision false ${batch} ${max_seq_length}
        done
    done
done
