#!/bin/bash
set -e
set -x
MAGICMIND_MODEL=$1
BATCH_SIZE=$2

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_python_output
mkdir -p $OUTPUT_DIR

echo "infer Magicmind model..."
python infer.py --device_id 0 \
                --magicmind_model $MAGICMIND_MODEL \
                --json_file $SQUAD_DATASETS_PATH/dev-v1.1.json \
                --batch_size ${BATCH_SIZE} \
                --max_seq_length 384 \
                --compute_accuracy true \
                --output_dir ${OUTPUT_DIR} \
                --acc_result ${OUTPUT_DIR}/acc_result.txt
