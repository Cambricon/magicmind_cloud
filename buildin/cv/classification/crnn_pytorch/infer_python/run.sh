#!/bin/bash

set -e
set -x
magicmind_model=${1}
batch_size=${2}
infer_res_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"
if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

if [ ! -d "${PROJ_ROOT_PATH}/data/output" ];
then
  mkdir "${PROJ_ROOT_PATH}/data/output"
fi

python infer.py  --valRoot ${SYNTH_DATASETS_PATH} \
                 --file_path ${SYNTH_DATASETS_PATH}/mnt/ramdisk/max/90kDICT32px/lexicon.txt \
                 --magicmind_model ${magicmind_model}\
                 --workers 8 \
                 --n_test_disp 40 \
                 --batch_size ${batch_size} \
                 --top1_file ${infer_res_dir}/infer_python_output_${batch_size}_log_eval
