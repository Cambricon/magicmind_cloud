#!/bin/bash
set -e
set -x

MM_RUN(){
    magicmind_model=$1
    batch_size=$2
    ${MM_RUN_PATH}/mm_run --magicmind_model $magicmind_model \
                          --iterations 1000 \
                          --batch_size ${batch_size} \
                          --devices 0
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh

for precision in force_float32 force_float16 qint8_mixed_float16
do
    for dynamic_shape in true false
    do
          for batch_size in 1 32 64
          do 
              magicmind_model=${MODEL_PATH}/yolov3_caffe_model_${precision}_${dynamic_shape}
              if [ ${dynamic_shape} == 'false' ];then
                  magicmind_model="${magicmind_model}_${batch_size}"
              fi

              # gen model
              if [ ! -f ${magicmind_model} ];then
                  cd ${PROJ_ROOT_PATH}/gen_model
                  bash run.sh ${magicmind_model} ${precision} ${batch_size} ${dynamic_shape}   
              else
                  echo "MagicMind model: ${magicmind_model} already exists!"
              fi

              # mm run
              MM_RUN ${magicmind_model} ${batch_size}
          done
    done
done