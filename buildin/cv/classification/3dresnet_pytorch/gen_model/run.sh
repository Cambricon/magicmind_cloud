#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}


python gen_model.py --precision ${precision} \
                    --batch_size ${batch_size} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${KINETICS_DATASETS_PATH}/kinetics_videos/jpg/ \
                    --input_dims ${batch_size} 3 16 112 112 \
                    --dim_range_min  1 3 16 112 112 \
                    --dim_range_max  64 3 16 112 112 \
                    --pytorch_pt ${MODEL_PATH}/3dresnet.pt \
                    --type64to32_conversion true \
                    --conv_scale_fold true






