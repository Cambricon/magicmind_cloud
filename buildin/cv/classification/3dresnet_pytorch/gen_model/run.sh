#!/bin/bash
set -e

PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
BATCH_SIZE=$2

if [ ! -f $PROJ_ROOT_PATH/data/models/3dresnet_${PRECISION}_${BATCH_SIZE}.mm ];then
    echo "generate Magicmind model begin..."
    if [ ! -d $PROJ_ROOT_PATH/data/models/ ];then
        mkdir -p $PROJ_ROOT_PATH/data/models/
    fi
    python gen_model.py \
        --pt_model $MODEL_PATH/3dresnet.pt \
        --output_model  $PROJ_ROOT_PATH/data/models/3dresnet_${PRECISION}_${BATCH_SIZE}.mm \
        --precision ${PRECISION} \
        --shape_mutable true \
        --batch_size ${BATCH_SIZE} \
	--image_dir $DATASETS_PATH/kinetics_videos/jpg/  
    echo "3dresnet.mm model saved in data/models/"
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/3dresnet_${PRECISION}_${BATCH_SIZE}.mm already exist."
fi
