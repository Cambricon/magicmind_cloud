#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}
img_h=224
img_w=224

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${img_h} ${img_w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${ILSVRC2012_DATASETS_PATH} \
                    --caffemodel ${MODEL_PATH}/mobilenet_v2.caffemodel \
                    --prototxt ${MODEL_PATH}/mobilenet_v2_deploy.prototxt \
                    --input_layout NHWC \
                    --means 103.939002991 116.778999329 123.680000305 \
                    --vars 3460.20761246 3460.20761246 3460.20761246 \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 64 3 ${img_h} ${img_w} \
                    --type64to32_conversion "true" \
                    --conv_scale_fold "true" 
