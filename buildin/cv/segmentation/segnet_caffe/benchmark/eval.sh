#!/bin/bash
set -e
set -x

COMPUTE_VOC_MIOU(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    WAYS=$4
    if [ ! -d $PROJ_ROOT_PATH/data/output/ ]; then mkdir -p $PROJ_ROOT_PATH/data/output/; fi 
    python $UTILS_PATH/compute_voc_mIOU_segnet.py --output_dir $PROJ_ROOT_PATH/data/output/${WAYS}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH} 2>&1 |tee $PROJ_ROOT_PATH/data/output/${WAYS}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH}_log_eval
}


cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do	
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision true $batch
        cd $PROJ_ROOT_PATH/infer_cpp
        bash run.sh $precision true $batch
        COMPUTE_VOC_MIOU $precision true $batch infer_cpp
    done
done
