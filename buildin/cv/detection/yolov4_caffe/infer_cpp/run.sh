#!/bin/bash
set -e
set -x

PRECISION=$1  
SHAPE_MUTABLE=$2
IMAGE_NUM=$3
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/"
    echo "mkdir sucessed!!!"
fi

if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1" ]; 
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1"
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $PROJ_ROOT_PATH/data/mm_model/yolov4_${PRECISION}_${SHAPE_MUTABLE}_1 \
                                  --image_dir $DATASETS_PATH/val2017 \
                                  --image_num ${IMAGE_NUM} \
                                  --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                  --label_path $UTILS_PATH/coco.names \
                                  --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1 \
                                  --save_img true
