#!/bin/bash

PRECISION=$1
BATCH_SIZE=$2
IMAGE_NUM=$3

if [ ! -d $PROJ_ROOT_PATH/data/images/${PRECISION}_${BATCH_SIZE} ];then
  mkdir -p $PROJ_ROOT_PATH/data/images/${PRECISION}_${BATCH_SIZE}
fi
bash build.sh
./bin/host_infer \
	--magicmind_model $PROJ_ROOT_PATH/data/models/arcface_${PRECISION}_${BATCH_SIZE}.mm \
    --image_dir $DATASETS_PATH/IJBC/loose_crop \
    --image_list  $DATASETS_PATH/IJBC/meta/ijbc_name_5pts_score.txt \
    --save_img true \
    --image_num $IMAGE_NUM \
    --output_dir $PROJ_ROOT_PATH/data/images/${PRECISION}_${BATCH_SIZE}

