#!/bin/bash
set -e
set -x
QUANT_MODE=$1
BATCH_SIZE=$2
THREADS=$3

MM_RUN(){
    QUANT_MODE=$1
    BATCH_SIZE=$2
    THREADS=$3
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/u2net_pytorch_${QUANT_MODE}_${BATCH_SIZE} \
                          --iterations 1000 \
			  --input_dims ${BATCH_SIZE},3,320,320 \
                          --threads ${THREADS} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${BATCH_SIZE}_log_perf
}

if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${BATCH_SIZE} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint16_mixed_float32
    do
        for batch in 1 4 8
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $quant_mode $batch 
            for thread in 1
            do 
                MM_RUN $quant_mode $batch $thread
                python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${batch}_log_perf --model u2net_pytorch
            done
        done
    done
fi

