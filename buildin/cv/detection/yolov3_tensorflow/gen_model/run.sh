#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}
img_h=416
img_w=416
input_names="input/input_data"

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size}  ${img_h} ${img_w} 3 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                    --tf_pb ${MODEL_PATH}/yolov3_coco_mmpost.pb \
                    --input_names ${input_names} \
                    --output_names "conv_sbbox/BiasAdd" "conv_mbbox/BiasAdd" "conv_lbbox/BiasAdd" \
                    --computation_preference fast   \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 32 3 ${img_h} ${img_w} 

