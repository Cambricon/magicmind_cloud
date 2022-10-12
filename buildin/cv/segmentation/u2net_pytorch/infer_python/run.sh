#!/bin/bash
QUANT_MODE=$1
BATCH_SIZE=$2
if [ ! -d $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${BATCH_SIZE} ];
then
    mkdir -p "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${BATCH_SIZE}"
fi
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/models/u2net_pytorch_${QUANT_MODE}_${BATCH_SIZE} \
                --img_dir $DATASETS_PATH \
                --output_folder $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${BATCH_SIZE} \
                --batch_size $BATCH_SIZE
