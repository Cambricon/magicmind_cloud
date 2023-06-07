#!/bin/bash
set -e
set -x

# export onnx model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

MM_RUN(){
    PRECISION=$1
    BATCH_SIZE=$2
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/ocrnet_${PRECISION}.mm \
                          --input_dims ${BATCH_SIZE},3,1024,2048 \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE}bs_log_perf
}

for precision in force_float32 force_float16 qint8_mixed_float16
do
    # generate magicmind model
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 
    # run perf
    for batch_size in 1 8
        do
            cd $PROJ_ROOT_PATH/gen_model
            MM_RUN $parameter_id $precision $batch_size
        done
done
