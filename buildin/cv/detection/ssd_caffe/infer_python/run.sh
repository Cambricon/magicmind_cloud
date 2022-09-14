#!/bin/bash
set -e
set -x

QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
BATCH=$4
SAVE_IMG=$5
if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
    echo "mkdir sucessed!!!"
fi
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/models/ssd_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                --devkit_path ${DATASETS_PATH}/VOCdevkit \
                --result_path $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/voc_preds \
                --batch ${BATCH} \
                --save_img ${SAVE_IMG}
