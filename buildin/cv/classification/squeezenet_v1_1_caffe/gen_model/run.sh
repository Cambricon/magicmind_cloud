#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}
img_h=227
img_w=227

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${img_h} ${img_w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${ILSVRC2012_DATASETS_PATH} \
                    --caffemodel ${MODEL_PATH}/squeezenet_v1_1.caffemodel \
                    --prototxt ${MODEL_PATH}/deploy_v1_1.prototxt \
                    --input_layout NHWC \
                    --means 104 117 123 \
                    --vars 1.0 1.0 1.0 \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 64 3 ${img_h} ${img_w} \
                    --type64to32_conversion "true" \
                    --conv_scale_fold "true" 

