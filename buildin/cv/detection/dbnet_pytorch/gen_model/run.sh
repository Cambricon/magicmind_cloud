#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}


python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 800 1280 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --dim_range_min 1 3 800 256 \
                    --dim_range_max 64 3 800 3072 \
                    --image_dir $TOTAL_TEXT_DATASETS_PATH/total_text/test_images \
                    --pytorch_pt $MODEL_PATH/dbnet.pt \
                    --mlu_arch mtp_372 \
                    --mean 123.675 116.28 103.53 \
                    --vars 65025 65025 65025 \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --input_layout NHWC




                    
                    
