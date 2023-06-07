#!/bin/bash
set -e
set -x
magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

PROTOTXT=googlenet_bn_deploy.prototxt
CAFFEMODEL=googlenet_bn.caffemodel

python gen_model.py  --precision  ${precision} \
                     --input_dims ${batch_size} 3 224 224 \
                     --dynamic_shape ${dynamic_shape} \
                     --magicmind_model  $magicmind_model \
                     --image_dir $ILSVRC2012_DATASETS_PATH \
                     --caffemodel  $MODEL_PATH/$CAFFEMODEL \
                     --prototxt $MODEL_PATH/$PROTOTXT \
                     --input_layout NHWC \
                     --means 103.939 116.779 123.68 \
                     --vars 1.0 1.0 1.0 \
                     --dim_range_min 1 3 224 224 \
                     --dim_range_max 64 3 224 224 \
                     --type64to32_conversion "true" \
                     --conv_scale_fold "true"  \
                     --device_id 0

