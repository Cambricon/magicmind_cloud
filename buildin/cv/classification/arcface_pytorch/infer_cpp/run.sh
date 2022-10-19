#!/bin/bash

QUANT_MODE=$1
BATCH_SIZE=$2

if [ ! -d $PROJ_ROOT_PATH/data/images/${QUANT_MODE}_${BATCH_SIZE} ];then
  mkdir -p $PROJ_ROOT_PATH/data/images/${QUANT_MODE}_${BATCH_SIZE}
fi
bash build.sh
./bin/host_infer \
	--magicmind_model $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
    --image_dir $DATASETS_PATH/IJBC/loose_crop \
    --image_list  $PROJ_ROOT_PATH/infer_cpp/1000_file_list \
    --save_img true \
    --output_dir $PROJ_ROOT_PATH/data/images/${QUANT_MODE}_${BATCH_SIZE}

