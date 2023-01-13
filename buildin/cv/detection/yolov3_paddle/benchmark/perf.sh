#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    if [ ! -d $PROJ_ROOT_PATH/data/output ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output"
    fi
    ${MM_RUN_PATH}/mm_run --magicmind_model $PROJ_ROOT_PATH/data/mm_model/yolov3_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
	                  --iterations 1000 \
			  --input_dims 1,2 ${BATCH_SIZE},3,608,608 1,2 \
			  --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
    qps=(`grep -wn "Throughput (qps):" $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf`)
    python $MAGICMIND_CLOUD/test/record_result.py --fps ${qps[2]}
}


###dynamic
echo "Parm Doesn't exist, run benchmark"
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for batch in 1 16 32
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision false $batch  
        MM_RUN $precision false $batch 
    done
done
