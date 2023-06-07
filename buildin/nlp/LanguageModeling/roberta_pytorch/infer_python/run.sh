#!/bin/bash
MAGICMIND_MODEL=${1}
BATCH_SIZE=${2:-1}
MAX_SEQ_LENGTH=${3:-128}

out_dir=${PROJ_ROOT_PATH}/data/output/$(basename ${MAGICMIND_MODEL})
OUT_DIR=${4:-${out_dir}}

if [ $# -lt 1 -o $# -gt 4 ];
then
    echo "bash run.sh <magicmind_model> <batch_size> <max_seq_len> <out_dir>"
    echo "Usage: bash run.sh ${MODEL_PATH}/roberta_force_float32_false_1_128 1 128 ${PROJ_ROOT_PATH}/data/output/fp32_1bs_128"
    exit -1
fi

if [ ! -d ${OUT_DIR} ];
then
    mkdir -p "${OUT_DIR}"
else 
    echo "folder: ${OUT_DIR} already exits"
fi

echo "infer Magicmind model..."
python infer.py --device_id 0 \
                --magicmind_model ${MAGICMIND_MODEL} \
                --batch_size ${BATCH_SIZE} \
                --max_seq_length ${MAX_SEQ_LENGTH} \
                --compute_accuracy true \
                --output_dir ${OUT_DIR} \
                --acc_result ${OUT_DIR}/acc_result.txt
