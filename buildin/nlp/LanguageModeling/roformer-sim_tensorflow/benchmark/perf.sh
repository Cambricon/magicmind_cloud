#!/bin/bash
set -e
set -x

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
MAX_SEQ_LEN=$4

MM_RUN(){
    PRECISION=$1 #force_float32/force_float16
    SHAPE_MUTABLE=$2 #true/false
    BATCH_SIZE=$3
    MAX_SEQ_LENGTH=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/roformer-sim_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --devices 0 \
			    2>&1 |tee $PROJ_ROOT_PATH/data/output/roformer-sim_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}_log_perf 
}

if [ $# != 0 ];
then
    MM_RUN ${PRECISION} ${SHAPE_MUTABLE} ${BATCH_SIZE} ${MAX_SEQ_LEN}
else
    echo "Parm Doesn't exist, run benchmark"
    for max_seq_length in 64
    do
        cd $PROJ_ROOT_PATH/export_model
        bash run.sh
        for precision in force_float32 force_float16
        do
            for batch in 1 16 32
            do 
                cd $PROJ_ROOT_PATH/gen_model
                bash run.sh $precision false $batch $max_seq_length
                
                MM_RUN $precision false $batch $max_seq_length 
                
            done
        done
    done
fi
