#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 416 416 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                    --input_layout NHWC \
                    --computation_preference fast   \
                    --dim_range_min 1 3 416 416 \
                    --dim_range_max 64 3 416 416 \
                    --means 0 0 0 \
                    --vars  65025 65025 65025 \
                    --prototxt ${MODEL_PATH}/yolov3_tiny.prototxt \
                    --caffemodel ${MODEL_PATH}/yolov3_tiny.caffemodel \
                    --type64to32_conversion true \
                    --conv_scale_fold true

