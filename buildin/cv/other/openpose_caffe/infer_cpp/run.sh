#!/bin/bash

PRECISION=$1
BATCH_SIZE=$2
bash ./build.sh
if [ ! -d $PROJ_ROOT_PATH/data/images/body25_${PRECISION}_${BATCH_SIZE} ];then
  mkdir -p $PROJ_ROOT_PATH/data/images/body25_${PRECISION}_${BATCH_SIZE}
fi
./bin/host_infer \
    --magicmind_model $MODEL_PATH/pose_body25_${PRECISION}_${BATCH_SIZE} \
    --image_dir $DATASETS_PATH \
    --image_list $PROJ_ROOT_PATH/infer_cpp/file_list \
    --save_img true \
    --output_dir $PROJ_ROOT_PATH/data/images/body25_${PRECISION}_${BATCH_SIZE} \
    --network BODY_25

if [ ! -d $PROJ_ROOT_PATH/data/images/coco_${PRECISION}_${BATCH_SIZE} ];then
  mkdir -p $PROJ_ROOT_PATH/data/images/coco_${PRECISION}_${BATCH_SIZE}
fi
./bin/host_infer \
    --magicmind_model $MODEL_PATH/pose_coco_${PRECISION}_${BATCH_SIZE} \
    --image_dir $DATASETS_PATH \
    --image_list $PROJ_ROOT_PATH/infer_cpp/file_list \
    --save_img true \
    --output_dir $PROJ_ROOT_PATH/data/images/coco_${PRECISION}_${BATCH_SIZE} \
    --network COCO
