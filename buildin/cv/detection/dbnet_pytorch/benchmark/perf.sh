#!/bin/bash
set -e
set -x

network=dbnet_pytorch

MM_RUN(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model ${magicmind_model}  \
                          --input_dims ${batch_size},800,1280,3 \
                          --devices 0 
}

cd ${PROJ_ROOT_PATH}/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
  for dynamic_shape in false #true
  do
    for batch_size in 1 16 32
    do
      magicmind_model=${MODEL_PATH}/${network}_model_${precision}_${dynamic_shape}_${batch_size}
      # gen model
      if [ ! -f ${magicmind_model} ];then
        cd ${PROJ_ROOT_PATH}/gen_model
        bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}
      else
          echo "MagicMind model: ${magicmind_model} already exists!"
      fi
      # run model
      MM_RUN ${magicmind_model} ${batch_size}
        
    done
  done
done
