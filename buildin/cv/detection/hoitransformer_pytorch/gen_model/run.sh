#!/bin/bash
set -e

PRECISION=$1 #force_float32/force_float16/
BATCH_SIZE=$2

if [ ! -f $PROJ_ROOT_PATH/data/models/hoitransformer_${PRECISION}_${BATCH_SIZE}.mm ];then
    echo "generate Magicmind model begin..."
    if [ ! -d $PROJ_ROOT_PATH/data/models/ ];then
        mkdir -p $PROJ_ROOT_PATH/data/models/
    fi
    python gen_model.py \
        --onnx_model $MODEL_PATH/hoitransformer.onnx \
        --mm_model  $PROJ_ROOT_PATH/data/models/hoitransformer_${PRECISION}_${BATCH_SIZE}.mm \
        --datasets_dir $HOIA_DATASETS_PATH/test  \
        --precision ${PRECISION} \
	--shape_mutable true \
        --batch_size ${BATCH_SIZE}
    echo "hoitransformer.mm model saved in data/models/"
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/hoitransformer_${PRECISION}_${BATCH_SIZE}.mm already exist."
fi
