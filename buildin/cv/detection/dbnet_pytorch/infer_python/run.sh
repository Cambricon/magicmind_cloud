#!/bin/bash
set -x
set -e

magicmind_model=${1}
batch_size=${2}

if grep -q "MMRunner" ${PROJ_ROOT_PATH}/export_model/DB/eval.py;
then
  echo "infer.patch already be used"
else
  patch -p0 ${PROJ_ROOT_PATH}/export_model/DB/eval.py < ${PROJ_ROOT_PATH}/infer_python/infer.patch
  patch -p0 ${PROJ_ROOT_PATH}/export_model/DB/data/processes/normalize_image.py < ${PROJ_ROOT_PATH}/infer_python/norm.patch
fi

if [ ! -d ${PROJ_ROOT_PATH}/data/output ];
then
    mkdir ${PROJ_ROOT_PATH}/data/output
fi
cd ${PROJ_ROOT_PATH}/export_model/DB
python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml  \
               --polygon \
               --box_thresh 0.7 \
               --magicmind_model ${magicmind_model} \
               --result_file ${PROJ_ROOT_PATH}/data/output/infer_python_output_${magicmind_model}_log_eval
