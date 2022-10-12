#!/bin/bash
set -e

QUANT_MODE=$1
BATCH_SIZE=$2
THREADS=$3

MM_RUN(){
    QUANT_MODE=$1
    BATCH_SIZE=$2
    THREADS=$3
        
    echo "Begin to inference in MLU device."
    $MM_RUN_PATH/mm_run \
        --magicmind_model $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --iterations 100 \
        --batch ${BATCH_SIZE} \
        --threads ${THREADS} 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${BATCH_SIZE}_log_perf
}

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi
if [ $# != 0 ];then
    MM_RUN ${QUANT_MODE} ${BATCH_SIZE} ${THREADS}
else
    quant_modes=('force_float32' 'force_float16' 'qint8_mixed_float16')
    batchs=(1 4 8)
    bash $PROJ_ROOT_PATH/export_model/run.sh
    for quant_mode in ${quant_modes[@]};
    do
      for batch in ${batchs[@]};
      do 
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode $batch
        for threads in 1  
        do
          echo "mm_run $quant_mode $batch $threads"
          MM_RUN $quant_mode $batch $threads
          python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${batch}_log_perf --model $MODEL_NAME
        done
      done
    done
fi

