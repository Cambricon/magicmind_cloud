#!/bin/bash
set -e
set -x

MM_RUN(){
    PARAMETER_ID=$1
    PRECISION=$2
    SHAPE_MUTABLE=$3
    BATCH_SIZE=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    if [ ${SHAPE_MUTABLE} == 'false' ];
    then
        MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID}
    else
        MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${PARAMETER_ID}
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MAGICMIND_MODEL \
                          --iterations 1000 \
                          --input_dims $BATCH_SIZE,320,256,1 \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID}_log_perf 
}



for parameter_id in 0
do
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh $parameter_id
    for precision in force_float32 force_float16 qint8_mixed_float16
    do
        for shape_mutable in false #true
        do
            for batch_size in 1 16 32
            do
                cd $PROJ_ROOT_PATH/gen_model
                bash run.sh $parameter_id $precision $shape_mutable $batch_size
                MM_RUN $parameter_id $precision $shape_mutable $batch_size
            done
        done
    done
done
