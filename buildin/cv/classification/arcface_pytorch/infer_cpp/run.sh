#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
image_num=${3}

infer_res_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"
if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

bash build.sh
./bin/host_infer --magicmind_model ${magicmind_model} \
                --batch_size ${batch_size} \
                --image_dir ${IJB_DATASETS_PATH}/IJBC/loose_crop \
                --image_list  ${IJB_DATASETS_PATH}/IJBC/meta/ijbc_name_5pts_score.txt \
                --save_img true \
                --image_num ${image_num} \
                --output_dir ${infer_res_dir}


