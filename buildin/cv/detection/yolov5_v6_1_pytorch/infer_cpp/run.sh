#!/bin/bash
set -e
set -x

QUANT_MODE=$1  
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
IMAGE_NUM=$4
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/"
    echo "mkdir sucessed!!!"
fi

if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1" ]; 
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1"
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $MODEL_PATH/yolov5_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                  --image_dir $DATASETS_PATH/val2017 \
                                  --image_num ${IMAGE_NUM} \
                                  --file_list $DATASETS_PATH/file_list_5000.txt \
                                  --label_path $DATASETS_PATH/coco.names \
                                  --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_1 \
                                  --save_img true
