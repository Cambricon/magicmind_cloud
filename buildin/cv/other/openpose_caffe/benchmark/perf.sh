#!/bin/bash
set -e
set -x

MM_RUN(){
   PRECISION=$1
   BATCH_SIZE=$2

   if [ ! -d $PROJ_ROOT_PATH/data/images ];
   then
       mkdir -p "$PROJ_ROOT_PATH/data/images"
   fi
   ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/pose_body25_${PRECISION}_${BATCH_SIZE} \
                         --iterations 1000 \
                         --batch_size ${BATCH_SIZE} \
                         --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/images/pose_body25_${PRECISION}_${BATCH_SIZE}_log_perf
   ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/pose_coco_${PRECISION}_${BATCH_SIZE} \
                         --iterations 1000 \
                         --batch_size ${BATCH_SIZE} \
                         --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/images/pose_coco_${PRECISION}_${BATCH_SIZE}_log_perf
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh 1
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for batch in 1 16 32
    do 
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision $batch
        MM_RUN $precision $batch
    done
done
