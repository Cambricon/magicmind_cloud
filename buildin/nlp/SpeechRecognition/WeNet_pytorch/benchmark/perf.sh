#!/bin/bash
set -e
set -x

MM_RUN_ENCODER(){
    PRECISION=$1
    BATCH_SIZE=$2
    SEQ_LEN=$3
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/encoder_${PRECISION}_model \
                          --iterations 100 \
                          --input_dims ${BATCH_SIZE},${SEQ_LEN},80 ${BATCH_SIZE} \
                          --devices 0 2>&1 | tee $PROJ_ROOT_PATH/data/output/encoder_${PRECISION}_${BATCH_SIZE}_${SEQ_LEN}_log_perf
}

MM_RUN_DECODER(){
    PRECISION=$1
    BATCH_SIZE=$2
    SEQ_LEN=$3
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/decoder_${PRECISION}_model \
                          --iterations 100 \
                          --input_dims ${BATCH_SIZE},`expr ${SEQ_LEN} / 4`,512 ${BATCH_SIZE} ${BATCH_SIZE},4,24 ${BATCH_SIZE},4 ${BATCH_SIZE},4\
                          --devices 0 2>&1 | tee $PROJ_ROOT_PATH/data/output/decoder_${PRECISION}_${BATCH_SIZE}_${SEQ_LEN}_log_perf
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

for precision in force_float32
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh ${precision}
    for batch in 1 16 32
    do
        for seqlen in 500
	do
            # mm run
            MM_RUN_ENCODER ${precision} ${batch} ${seqlen}  
            MM_RUN_DECODER ${precision} ${batch} ${seqlen}  
        done
    done
    
done

