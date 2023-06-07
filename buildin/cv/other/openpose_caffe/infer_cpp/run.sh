#!/bin/bash
set -x
magicmind_model=${1}
batch_size=${2}
image_num=${3}
backbone=${4:-COCO}
tmp_name=$(basename ${magicmind_model})
infer_res_dir="${PROJ_ROOT_PATH}/data/output/${tmp_name}_infer_cpp_res"

if [ ! -d ${infer_res_dir} ];then
  mkdir -p ${infer_res_dir}
fi

cd ${PROJ_ROOT_PATH}/infer_cpp/
if [ -f infer ];then
rm infer
fi
    bash build.sh
#if [ ! -f infer ];then
#    bash build.sh
#fi

        #--image_list /home/develop_0406/magicmind_cloud/buildin/cv/other/openpose_caffe/test_coco_file_list_1.txt \
        #--image_list ${UTILS_PATH}//home/develop_0406/magicmind_cloud/buildin/cv/utils/coco_file_list_5000.txt
./infer --magicmind_model ${magicmind_model} \
        --batch_size ${batch_size} \
        --image_dir ${COCO_DATASETS_PATH}/val2017 \
        --image_list ${UTILS_PATH}/coco_file_list_5000.txt \
        --output_dir ${infer_res_dir} \
        --image_num ${image_num} \
        --network ${backbone} 

# get metric res
function compute_coco(){
    infer_res_dir=${1}
    log_file=${infer_res_dir}/log_eval
	ann_file=${COCO_DATASETS_PATH}/annotations/person_keypoints_val2017.json
	res_file=${infer_res_dir}/${backbone}
	output_file=${infer_res_dir}/metric_res_file
    python ${UTILS_PATH}/compute_coco_keypoints.py --ann_file ${ann_file} \
                                           --res_file ${res_file} \
                                           --output_file ${output_file} 2>&1 |tee ${log_file}

}

compute_coco  ${infer_res_dir} 

