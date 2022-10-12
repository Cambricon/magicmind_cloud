#!/bin/bash
set -e
set -x

QUANT_MODE=$1
SHAPE_MUTABLE=$2
IMAGE_NUM=$3
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}"
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MODEL_PATH/deeplabv3_tf_model_${QUANT_MODE}_${SHAPE_MUTABLE} \
	                        --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages/ \
                                --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE} \
                                --file_list $DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                                --image_num ${IMAGE_NUM} \
				--shape_mutable ${SHAPE_MUTABLE} \
				--save_img true
