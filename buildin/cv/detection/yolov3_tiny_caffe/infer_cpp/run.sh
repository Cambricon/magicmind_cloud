#!/bin/bash
set -e
set -x

magicmind_model=${1}
batch_size=${2}
image_num=${3}

json_name="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model}).json"
infer_res_dir="${PROJ_ROOT_PATH}/data/output/$(basename ${magicmind_model})_infer_res"
if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

bash build.sh
${PROJ_ROOT_PATH}/infer_cpp/infer   --magicmind_model ${magicmind_model} \
                                    --image_num ${image_num}\
                                    --batch_size ${batch_size}\
                                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                                    --file_list ${UTILS_PATH}/coco_file_list_5000.txt \
                                    --label_path ${UTILS_PATH}/coco.names \
                                    --output_dir ${infer_res_dir}\
                                    --save_img true

# get metric res
function compute_coco(){
    infer_res_dir=${1}
    json_name=${2}
    log_file=${infer_res_dir}/log_eval
    python ${UTILS_PATH}/compute_coco_mAP.py  --file_list ${UTILS_PATH}/coco_file_list_5000.txt \
                                              --result_dir ${infer_res_dir} \
                                              --ann_dir ${COCO_DATASETS_PATH} \
                                              --data_type val2017 \
                                              --json_name ${json_name} \
                                              --infer_mode infer_cpp \
                                              --image_num ${image_num} 2>&1 |tee ${log_file}
}

compute_coco  ${infer_res_dir} ${json_name}

