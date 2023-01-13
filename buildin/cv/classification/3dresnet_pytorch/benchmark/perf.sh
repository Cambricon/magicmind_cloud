#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    BATCH_SIZE=$2
    ${MM_RUN_PATH}/mm_run   --magicmind_model $MODEL_PATH/3dresnet_${PRECISION}_${BATCH_SIZE}.mm \
                            --iterations 1000 \
                            --batch_size ${BATCH_SIZE} \
                            --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE}_log_perf
}


cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do  
    for batch_size in 1 32 64
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision $batch_size
        MM_RUN $precision $batch_size
    done
done
