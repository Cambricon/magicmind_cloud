#!/bin/bash
PRECISION=$1
BATCH_SIZE=$2
if [ ! -d $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${BATCH_SIZE} ];
then
    mkdir -p "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${BATCH_SIZE}"
fi
python infer.py --magicmind_model $MODEL_PATH/u2net_pytorch_${PRECISION}_${BATCH_SIZE} \
                --img_dir $MSRA_B_DATASETS_PATH \
                --output_folder $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${BATCH_SIZE} \
                --batch_size $BATCH_SIZE
