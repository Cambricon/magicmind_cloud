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

bash build.sh
${PROJ_ROOT_PATH}/infer_cpp/infer   --magicmind_model ${magicmind_model} \
                                  --image_dir ${WIDERFACE_DATASETS_PATH}/WIDER_val/images \
                                  --image_num -1 \
                                  --file_list ${PROJ_ROOT_PATH}/infer_cpp/wider_val.txt \
                                  --output_dir ${infer_res_dir} \
                                  --save_img true \
                                  --batch_size ${batch_size}

# get metrics res
cd ${PROJ_ROOT_PATH}/export_model/Pytorch_Retinaface/widerface_evaluate
python3 setup.py build_ext --inplace
python3 evaluation.py -p ${infer_res_dir}/pred_txts \
                      -g ${PROJ_ROOT_PATH}/export_model/Pytorch_Retinaface/widerface_evaluate/ground_truth
