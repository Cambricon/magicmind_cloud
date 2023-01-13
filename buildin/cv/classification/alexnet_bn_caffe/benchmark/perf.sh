#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    if [ ${SHAPE_MUTABLE} == 'false' ];
    then
        MAGICMIND_MODEL=$MODEL_PATH/alexnet_bn_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    else
        MAGICMIND_MODEL=$MODEL_PATH/alexnet_bn_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
    fi
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MAGICMIND_MODEL \
                          --iterations 1000 \
                          --batch_size ${BATCH_SIZE} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}


cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
  for shape_mutable in false
  do
    for batch in 1 32 64
    do 
      cd $PROJ_ROOT_PATH/gen_model
      bash run.sh $precision $shape_mutable $batch
      MM_RUN $precision $shape_mutable $batch
    done
  done
done

