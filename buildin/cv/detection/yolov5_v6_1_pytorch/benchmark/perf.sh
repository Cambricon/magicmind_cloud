#!/bin/bash
set -e
set -x

MM_RUN(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    ${MM_RUN_PATH}/mm_run   --magicmind_model $MODEL_PATH/yolov5_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                            --iterations 1000 \
                            --batch_size ${BATCH_SIZE} \
                            --devices 0 2>&1 |tee $PROJ_ROOT_PATH/data/output/${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_log_perf
}

languages=infer_cpp
image_num=5000
conf=0.001
iou=0.65
max_det=1000

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for shape_mutable in false
do
    for precision in force_float32 force_float16 qint8_mixed_float16
    do  
        for batch_size in 1 4 8
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $precision $shape_mutable $batch_size $conf $iou $max_det
            MM_RUN $precision $shape_mutable $batch_size
        done
    done
done