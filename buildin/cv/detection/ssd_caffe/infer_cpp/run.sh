#!/bin/bash
set -e
set -x

PRECISION=$1  
SHAPE_MUTABLE=$2
SAVE_IMG=$3

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1
if [ -d $OUTPUT_DIR ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir $OUTPUT_DIR
    mkdir $OUTPUT_DIR/voc_preds
    echo "mkdir successed!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/ssd_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_1
else
    MAGICMIND_MODEL=$MODEL_PATH/ssd_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi

bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MAGICMIND_MODEL \
                                --image_dir $VOC2007_DATASETS_PATH/VOCdevkit \
                                --output_dir $OUTPUT_DIR \
                                --save_img $SAVE_IMG
