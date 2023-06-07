#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
image_num=${3}

OUTPUT_DIR="${MODEL_PATH}/../output/$(basename ${magicmind_model})"
mkdir -p ${OUTPUT_DIR}

python infer.py --device_id 0 \
                --magicmind_model ${magicmind_model} \
                --image_dir ${ILSVRC2012_DATASETS_PATH} \
                --image_num ${image_num} \
                --name_file ${UTILS_PATH}/imagenet_name.txt \
                --label_file ${UTILS_PATH}/ILSVRC2012_val.txt \
                --result_file ${OUTPUT_DIR}/infer_result.txt \
                --result_label_file ${OUTPUT_DIR}/eval_labels.txt \
                --result_top1_file ${OUTPUT_DIR}/eval_result_1.txt \
                --result_top5_file ${OUTPUT_DIR}/eval_result_5.txt \
                --batch_size ${batch_size}
                
python ${UTILS_PATH}/compute_top1_and_top5.py   --result_label_file ${OUTPUT_DIR}/eval_labels.txt \
                                                --result_1_file ${OUTPUT_DIR}/eval_result_1.txt \
                                                --result_5_file ${OUTPUT_DIR}/eval_result_5.txt \
                                                --top1andtop5_file ${OUTPUT_DIR}/eval_result.txt

