#!/bin/bash
set -x
magicmind_model=${1}
batch_size=${2}
image_num=${3}
tmp_name=$(basename ${magicmind_model})
infer_res_dir="${PROJ_ROOT_PATH}/data/output/${tmp_name}_infer_cpp_res"

if [ ! -d ${infer_res_dir} ];
then
  mkdir -p ${infer_res_dir}
fi

cd $PROJ_ROOT_PATH/infer_cpp/
if [ ! -f infer ];then
    bash build.sh
fi

video_list="${UCF101_DATASETS_PATH}/ucfTrainTestlist/testlist01.txt" 
#testlist01.txt 包含3783条视频 ，若对全部视频进行推理，则时间较长 请耐心等待

./infer --magicmind_model ${magicmind_model} \
        --batch_size ${batch_size} \
        --video_list ${video_list} \
        --image_dir ${UCF101_DATASETS_PATH} \
        --name_file ${UCF101_DATASETS_PATH}/ucfTrainTestlist/classInd.txt \
        --result_file ${infer_res_dir}/infer_result.txt \
        --result_label_file ${infer_res_dir}/eval_labels.txt \
        --result_top1_file ${infer_res_dir}/eval_result_1.txt \
        --result_top5_file ${infer_res_dir}/eval_result_5.txt \
        --image_num ${image_num} 


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

