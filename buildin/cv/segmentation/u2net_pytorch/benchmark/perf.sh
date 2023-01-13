#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    BATCH_SIZE=$2
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/u2net_pytorch_${PRECISION}_${BATCH_SIZE} \
                          --iterations 1000 \
			  --input_dims ${BATCH_SIZE},3,320,320 \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE}_log_perf
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for batch in 1 16 32
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision $batch 
        MM_RUN $precision $batch
    done
done
