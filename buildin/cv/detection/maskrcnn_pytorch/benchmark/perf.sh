#!/bin/bash
set -e
set -x

MM_RUN(){
    QUANT_MODE=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    THREADS=$4
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/mm_model/${QUANT_MODE}_${SHAPE_MUTABLE}_1 \
                          --iterations 1000 \
                          --threads ${THREADS} \
                          --input_dims ${BATCH_SIZE},3,800,800 \
                          --devices 0 2>&1 | tee $PROJ_ROOT_PATH/data/output/${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}

#dynamic
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
cd $PROJ_ROOT_PATH/gen_model/
if [ ! -d $PROJ_ROOT_PATH/data/output/ ];then
    mkdir -p $PROJ_ROOT_PATH/data/output/
fi

for quant_mode in force_float32
do
    for batch in 1 4 8
    do
        for shape_mutable in true
        do
            MM_MODEL=${quant_mode}_${shape_mutable}_1
            if [ ! -f $PROJ_ROOT_PATH/data/mm_model/$MM_MODEL ];then
                bash run.sh $quant_mode $shape_mutable $batch
            fi
            for threads in 1  
            do
                # mm run
                MM_RUN $quant_mode $shape_mutable $batch $threads
                # compare perf
                python $MAGICMIND_CLOUD/test/compare_perf.py    --output_file $PROJ_ROOT_PATH/data/output/${quant_mode}_${shape_mutable}_${batch}_log_perf \
                                                                --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${shape_mutable}_${batch}_log_perf \
                                                                --model mask_rcnn
            done
        done
      done
done
