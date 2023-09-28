#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
sample_num=${3}

OUTPUT_DIR="${MODEL_PATH}/../output/$(basename ${magicmind_model})"
mkdir -p ${OUTPUT_DIR}

python infer.py --device_id 0 \
                --magicmind_model ${magicmind_model} \
                --dataset_dir ${CRITEO_DATASETS_PATH}/slot_test_data_full \
                --sample_num ${sample_num} \
                --batch_size ${batch_size} \
                --result_file ${OUTPUT_DIR}/infer_result.txt
