#!/bin/bash
set -e
set -x

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
THREADS=$4

MM_RUN(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    THREADS=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                          --iterations 1000 \
                          --batch ${BATCH_SIZE} \
                          --threads ${THREADS} \
                          --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}

###dynamic
if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH_SIZE} ${BATCH_SIZE} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
      mkdir -p $PROJ_ROOT_PATH/data/output
    fi
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
      for batch in 1 4 8
      do 
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true $batch
        for thread in 1  
        do
          MM_RUN $quant_mode true $batch $thread
	  python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_true_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_true_${batch}_log_perf --model ${MODEL_NAME}
        done
      done
    done
fi
####static
#for quant_mode in force_float32 force_float16 qint8_mixed_float16
#do
#  for batch in 1 4 8
#  do
#    cd $PROJ_ROOT_PATH/gen_model
#    bash run.sh $quant_mode false $batch
#    for thread in 1
#    do
#      MM_RUN $quant_mode false $batch $batch $thread
#    done
#  done
#done
#
