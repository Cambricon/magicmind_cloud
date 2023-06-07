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

python infer.py  --device_id 0 \
                 --magicmind_model ${magicmind_model} \
                 --image_dir ${ILSVRC2012_DATASETS_PATH} \
                 --image_num ${image_num} \
                 --name_file ${UTILS_PATH}/imagenet_name.txt \
                 --label_file ${UTILS_PATH}/ILSVRC2012_val.txt \
                 --result_file ${infer_res_dir}/infer_result.txt \
                 --result_label_file ${infer_res_dir}/eval_labels.txt \
                 --result_top1_file ${infer_res_dir}/eval_result_1.txt \
                 --result_top5_file ${infer_res_dir}/eval_result_5.txt \
                 --batch_size ${batch_size}

compute_top1_and_top5(){
    infer_res_dir=${1}
    python $UTILS_PATH/compute_top1_and_top5.py --result_label_file ${infer_res_dir}/eval_labels.txt \
                                                --result_1_file ${infer_res_dir}/eval_result_1.txt \
                                                --result_5_file ${infer_res_dir}/eval_result_5.txt \
                                                --top1andtop5_file ${infer_res_dir}/eval_result.txt
}

compute_top1_and_top5 ${infer_res_dir}

