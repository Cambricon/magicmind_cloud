#!/bin/bash
set -e

#QUANT_MODE=qint8_mixed_float16
QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
BATCH_SIZE=$2
#BATCH_SIZE=128

if [ ! -f $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm ];then
    echo "generate Magicmind model begin..."
    if [ ! -d $PROJ_ROOT_PATH/data/models/ ];then
	mkdir -p $PROJ_ROOT_PATH/data/models/
    fi
    python gen_model.py \
        --pt_model $MODEL_PATH/arcface_r100.pt \
        --output_model_path  $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm \
        --image_dir ./file_list.txt  \
        --quant_mode ${QUANT_MODE} \
        --batch_size ${BATCH_SIZE} 
    echo "arcface.mm model saved in data/models/"
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/arcface_${QUANT_MODE}_${BATCH_SIZE}.mm already exist."
fi
    
