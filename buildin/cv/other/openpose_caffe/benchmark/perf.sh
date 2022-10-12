#!/bin/bash
set -e
set -x
QUANT_MODE=$1
BATCH_SIZE=$2
THREADS=$3

MM_RUN(){
   QUANT_MODE=$1
   BATCH_SIZE=$2
   THREADS=$3

   if [ ! -d $PROJ_ROOT_PATH/data/images ];
    then
        mkdir "$PROJ_ROOT_PATH/data/images"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/pose_body25_${QUANT_MODE}_${BATCH_SIZE} \
	                  --iterations 1000 \
			  --batch ${BATCH_SIZE} \
			  --threads ${THREADS} \
			  --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/images/pose_body25_${QUANT_MODE}_${BATCH_SIZE}_log_perf
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/models/pose_coco_${QUANT_MODE}_${BATCH_SIZE} \
	                  --iterations 1000 \
			  --batch ${BATCH_SIZE} \
			  --threads ${THREADS} \
			  --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/images/pose_coco_${QUANT_MODE}_${BATCH_SIZE}_log_perf
}

if [ $# != 0 ];
then
    MM_RUN ${QUANT_MODE} ${BATCH_SIZE} ${THREADS}
else
    echo "Parm Doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1
    for quant_mode in force_float32 force_float16 qint8_mixed_float16
    do
        for batch in 1 4 8
        do 
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $quant_mode $batch
            for threads in 1  
            do
                MM_RUN $quant_mode $batch $threads
                python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/images/pose_body25_${quant_mode}_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/pose_body25_${quant_mode}_${batch}_log_perf --model pose_body25
                python $MAGICMIND_CLOUD/test/compare_perf.py --output_file $PROJ_ROOT_PATH/data/images/pose_coco_${quant_mode}_${batch}_log_perf --output_ok_file $PROJ_ROOT_PATH/data/output_ok/pose_coco_${quant_mode}_${batch}_log_perf --model pose_coco
            done
        done
    done
fi

