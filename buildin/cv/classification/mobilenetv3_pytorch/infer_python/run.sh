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

python infer.py --magicmind_model ${magicmind_model} \
                --image_num ${image_num} \
                --batch_size ${batch_size} \
                --image_dir ${ILSVRC2012_DATASETS_PATH} \
                --label_file ${UTILS_PATH}/ILSVRC2012_val.txt \
                --name_file ${UTILS_PATH}/imagenet_name.txt \
                --result_file ${infer_res_dir}/infer_result.txt \
                --result_label_file ${infer_res_dir}/eval_labels.txt \
                --result_top1_file ${infer_res_dir}/eval_result_1.txt \
                --result_top5_file ${infer_res_dir}/eval_result_5.txt 


# get metric res
function compute_acc(){
    infer_res_dir=${1}
    log_file=${infer_res_dir}/log_eval
    python ${UTILS_PATH}/compute_top1_and_top5.py \
            --result_label_file ${infer_res_dir}/eval_labels.txt \
            --result_1_file ${infer_res_dir}/eval_result_1.txt \
            --result_5_file ${infer_res_dir}/eval_result_5.txt \
            --top1andtop5_file ${infer_res_dir}/eval_result.txt 2>&1 |tee ${log_file}

}

compute_acc  ${infer_res_dir} 

