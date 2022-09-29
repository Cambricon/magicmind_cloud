#!/bin/bash
set -e
set -x

QUANT_MODE=$1  
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1"
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi

bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MODEL_PATH/segnet_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages/ \
				--image_list $DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                                --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1 \
                                --save_txt true
