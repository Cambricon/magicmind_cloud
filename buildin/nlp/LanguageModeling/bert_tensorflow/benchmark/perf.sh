#!/bin/bash
set -e
set -x

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
MAX_SEQ_LEN=$4
THREADS=$5

MM_RUN(){
    QUANT_MODE=$1 #force_float32/force_float16
    SHAPE_MUTABLE=$2 #true/false
    BATCH_SIZE=$3
    MAX_SEQ_LENGTH=$4
    THREADS=$5
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/bert_tensorflow_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                          --iterations 1000 \
                          --input_dims ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} ${BATCH_SIZE},${MAX_SEQ_LENGTH} \
                          --threads ${THREADS} \
                          --devices 0 \
			  --input_files $PROJ_ROOT_PATH/data/input_0_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input_1_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin $PROJ_ROOT_PATH/data/input_2_${BATCH_SIZE}_${MAX_SEQ_LENGTH}.bin 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH}_log_perf 
}

if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH_SIZE} ${MAX_SEQ_LEN} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    for max_seq_length in 384
    do
        cd $PROJ_ROOT_PATH/export_model
        bash run.sh
        for quant_mode in force_float32 force_float16
        do
            for batch in 1 4 8
            do 
                cd $PROJ_ROOT_PATH/gen_model
                bash run.sh $quant_mode true $batch $max_seq_length
                for thread in 1  
                do
                    MM_RUN $quant_mode true $batch $max_seq_length $thread
                    python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_true_${batch}_${max_seq_length}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_true_${batch}_${max_seq_length}_log_perf --model bert_tensorflow
                done
            done
        done
    done
fi
