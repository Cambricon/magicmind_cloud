#!/bin/bash
set -e
set -x

COMPUTE_VOC(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    WAYS=$3
    python $UTILS_PATH/compute_voc_mAP.py --path $PROJ_ROOT_PATH/data/output/${WAYS}_output_${PRECISION}_${SHAPE_MUTABLE}_1/voc_preds/ \
                                          --devkit_path $VOC2007_DATASETS_PATH/VOCdevkit 2>&1 |tee $PROJ_ROOT_PATH/data/output/${WAYS}_output_${PRECISION}_${SHAPE_MUTABLE}_1/voc_preds/log_eval
}
 
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do 
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision true 1
    cd $PROJ_ROOT_PATH/infer_python
    bash run.sh $precision true true
    COMPUTE_VOC $precision true infer_python
done
