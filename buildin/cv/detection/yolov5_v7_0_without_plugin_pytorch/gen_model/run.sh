#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 640 640 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                    --pytorch_pt ${MODEL_PATH}/yolov5s_traced.pt \
                    --input_layout NHWC \
                    --computation_preference fast   \
                    --type64to32_conversion true   \
                    --dim_range_min 1 3 128 128 \
                    --dim_range_max 64 3 960 960 \
                    --means 0 0 0 \
                    --vars  65025 65025 65025 

