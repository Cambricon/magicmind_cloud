#!/bin/bash
set -e
set -x
QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
BATCH=$4
THREADS=$5

MM_RUN(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    BATCH=$4
    THREADS=$5
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/retinaface_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
	                  --iterations 1000 \
			  --batch ${BATCH} \
			  --threads ${THREADS} \
			  --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}_log_perf
    fps=`cat "$PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}_log_perf" | grep "Throughput (qps):"`
    fps=`eval echo "${fps:18}"`
    python $MAGICMIND_CLOUD/test/record_result.py --fps ${fps}
}

###dynamic
if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${SHAPE_MUTABLE} ${BATCH_SIZE} ${BATCH} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode true 1 0.001 0.65 1000
        for batch in 1 4 8
        do 
            for threads in 1  
            do
                MM_RUN $quant_mode true 1 $batch $threads
                python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_true_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_true_${batch}_log_perf --model retinaface_pytorch
            done
        done
    done
fi




