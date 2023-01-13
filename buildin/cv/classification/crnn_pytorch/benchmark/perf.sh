#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    BATCH=$2
    W=$3
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $MODEL_PATH/crnn_pt_model_${PRECISION} \
                          --iterations 1000 \
                          --input_dims ${BATCH},1,32,${W} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${BATCH}_1_32_${W}_log_perf
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16
do
  cd $PROJ_ROOT_PATH/gen_model
  bash run.sh $precision
  for batch in 1 16 32
  do 
    for w in 200
    do 
      MM_RUN $precision $batch $thread $w
    done
  done
done
