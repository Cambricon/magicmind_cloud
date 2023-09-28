#!/bin/bash
set -e
set -x

PRECISION=$1
IMAGE_NUM=$2
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir $PROJ_ROOT_PATH/data/output
fi
if [ ! -d $PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION} ];
then
    mkdir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MODEL_PATH/deeplabv3_tensorflow_model_${PRECISION} \
                                --image_dir $VOC2012_DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages \
                                --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION} \
                                --file_list $VOC2012_DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                                --image_num $IMAGE_NUM \
                                --save_img true
