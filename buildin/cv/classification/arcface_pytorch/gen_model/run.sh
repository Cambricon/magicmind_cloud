#!/bin/bash
set -e
set -x 

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}


python gen_model.py --pytorch_pt ${MODEL_PATH}/arcface_r100.pt \
                    --magicmind_model  ${magicmind_model} \
                    --image_dir ${PROJ_ROOT_PATH}/gen_model/file_list.txt  \
                    --precision ${precision} \
                    --input_dims ${batch_size} 3 112 112 \
                    --dynamic_shape ${dynamic_shape} \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --mlu_arch mtp_372 \
                    --input_layout NHWC \
                    --dim_range_min 1 3 112 112 \
                    --dim_range_max 256 3 112 112 \
                    --means 127.5 127.5 127.5 \
                    --vars  16256.25 16256.25 16256.25 



    
