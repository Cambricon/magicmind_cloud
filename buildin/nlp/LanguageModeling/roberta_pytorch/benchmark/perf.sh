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
    if [ ${SHAPE_MUTABLE} == 'false' ];
    then
      MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/roberta_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}_model
    else
      MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/roberta_${PRECISION}_${SHAPE_MUTABLE}_${MAX_SEQ_LENGTH}_model
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MAGICMIND_MODEL \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --devices 0 \
			  --input_files $PROJ_ROOT_PATH/data/input0_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input1_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input2_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}


for max_seq_length in 128
do
    for batch in 1 16 32
    do
        cd $PROJ_ROOT_PATH/export_model
        bash run.sh $batch $max_seq_length
        for precision in force_float32 force_float16
        do
           cd $PROJ_ROOT_PATH/gen_model
           bash run.sh $precision false $batch $max_seq_length
           MM_RUN $precision false $batch $max_seq_length
	done
    done
done
