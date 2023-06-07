#!/bin/bash
set -x
magicmind_model=${1}
batch_size=${2}
image_num=${3:-0}
tmp_name=$(basename ${magicmind_model})
infer_res_dir="${PROJ_ROOT_PATH}/data/output/${tmp_name}_infer_cpp_res"

if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

cd ${PROJ_ROOT_PATH}/infer_cpp/
if [ ! -f infer ];then
    bash build.sh
fi

cur_path=$(pwd)
./infer --magicmind_model ${magicmind_model} \
        --batch_size ${batch_size} \
        --image_dir ${VOC2012_DATASETS_PATH}/VOCdevkit/VOC2012/JPEGImages \
        --image_list ${VOC2012_DATASETS_PATH}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
        --output_dir ${infer_res_dir} \
        --image_num ${image_num} 


# get metric res
function compute_voc_miou(){
    infer_res_dir=${1}
    log_file=${infer_res_dir}/log_eval
    python ${UTILS_PATH}/compute_voc_mIOU_segnet.py \
             --output_dir ${infer_res_dir} 2>&1 |tee ${log_file}

}

compute_voc_miou  ${infer_res_dir} 

