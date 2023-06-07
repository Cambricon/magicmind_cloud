#!/bin/bash
set -e
set -x

network=paddle_ocr

MM_RUN(){
    magicmind_det_model=$1
    magicmind_rec_model=$2
    magicmind_cls_model=$3
    batch_size=$4
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_det_model}  \
                          --input_dims ${batch_size},3,704,1280 \
                          --devices 0 
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_rec_model}  \
                          --input_dims ${batch_size},3,48,320 \
                          --devices 0
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_cls_model}  \
                          --input_dims ${batch_size},3,48,320 \
                          --devices 0 
}

cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
for precision in force_float32 force_float16
do
  for dynamic_shape in false #true
  do
    for batch_size in 1 16 32
    do
      magicmind_det_model=${MODEL_PATH}/${network}_det_model_${precision}_${dynamic_shape}
      if [ ${dynamic_shape} == 'false' ];then
          magicmind_det_model="${magicmind_det_model}_${batch_size}"
      fi
      magicmind_rec_model=${MODEL_PATH}/${network}_rec_model_${precision}_${dynamic_shape}
      if [ ${dynamic_shape} == 'false' ];then
          magicmind_rec_model="${magicmind_rec_model}_${batch_size}"
      fi
      magicmind_cls_model=${MODEL_PATH}/${network}_cls_model_${precision}_${dynamic_shape}
      if [ ${dynamic_shape} == 'false' ];then
          magicmind_cls_model="${magicmind_cls_model}_${batch_size}"
      fi
      ### 1.gen model
      if [ ! -f ${magicmind_det_model} ] || [ ! -f ${magicmind_rec_model} ] || [ ! -f ${magicmind_cls_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_det_model} ${magicmind_rec_model} ${magicmind_cls_model} ${precision} ${batch_size} ${dynamic_shape}
      else
        echo "MagicMind model: ${magicmind_det_model} , ${magicmind_rec_model} and ${magicmind_cls_model} already exists!"
      fi
      # run model
      MM_RUN ${magicmind_det_model} ${magicmind_rec_model} ${magicmind_cls_model} ${batch_size}        
    done
  done
done