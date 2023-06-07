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
                    --image_dir ${VOC2012_DATASETS_PATH}/VOCdevkit/VOC2012/JPEGImages \
                    --caffemodel ${MODEL_PATH}/segnet_pascal.caffemodel \
                    --prototxt ${MODEL_PATH}/segnet_pascal.prototxt \
                    --input_layout NHWC \
                    --output_layout NHWC \
                    --means 0 0 0 \
                    --vars 1.0 1.0 1.0 \
                    --cluster_num 6 8 \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 32 3 ${img_h} ${img_w} \
                    --type64to32_conversion "true" \
                    --weight_quant_granularity "per_axis" \
                    --conv_scale_fold "true" 

