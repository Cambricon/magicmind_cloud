#!/bin/bash
set -e

PRECISION=$1
BATCH_SIZE=$2

MM_RUN(){
    PRECISION=$1
    BATCH_SIZE=$2
        
    echo "Begin to inference in MLU device."
    $MM_RUN_PATH/mm_run \
        --magicmind_model $PROJ_ROOT_PATH/data/models/arcface_${PRECISION}_${BATCH_SIZE}.mm \
        --iterations 100 \
        --batch_size ${BATCH_SIZE} 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH_SIZE}_log_perf
}

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi
precisions=('force_float32' 'force_float16' 'qint8_mixed_float16')
batchs=(1 32 64)
bash $PROJ_ROOT_PATH/export_model/run.sh
for precision in ${precisions[@]};
do
  for batch in ${batchs[@]};
  do 
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision $batch
    echo "mm_run $precision $batch"
    MM_RUN $precision $batch
  done
done
