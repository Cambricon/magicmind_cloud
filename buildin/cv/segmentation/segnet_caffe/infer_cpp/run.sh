#!/bin/bash
set -e
set -x
PRECISION=$1  
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
if [ ! -d $OUTPUT_DIR ];
then
    mkdir $OUTPUT_DIR
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/segnet_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/segnet_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MAGICMIND_MODEL \
                                --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages/ \
                                --image_list $DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                                --output_dir $OUTPUT_DIR \
                                --save_txt true
