#!/bin/bash

PRECISION=$1
BATCH=$2
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
fi

python infer.py  --valRoot $DATASETS_PATH \
                 --file_path $DATASETS_PATH/mnt/ramdisk/max/90kDICT32px/lexicon.txt \
                 --magicmind_model $MODEL_PATH/crnn_pt_model_${PRECISION} \
                 --workers 8 \
                 --n_test_disp 40 \
                 --batchSize ${BATCH} \
                 --top1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${BATCH}_log_eval