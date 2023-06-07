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
                    --pytorch_pt ${MODEL_PATH}/mobilenet-v3_small.torchscript.pt \
                    --input_layout NHWC \
                    --means 123.675 116.28 103.53 \
                    --vars 3409.976025 3262.6944 3291.890625 \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 64 3 ${img_h} ${img_w} \
                    --type64to32_conversion "true" \
                    --conv_scale_fold "true" 
