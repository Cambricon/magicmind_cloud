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
                    --dim_range_min 1 3 416 416 \
                    --dim_range_max 64 3 416 416 \
                    --prototxt ${MODEL_PATH}/yolov4.prototxt \
                    --caffemodel ${MODEL_PATH}/yolov4.caffemodel \
                    --type64to32_conversion true \
                    --conv_scale_fold true

