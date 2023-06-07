#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}
img_h=368
img_w=656

# use soft-link (openpose.caffemodel, openpose.prototxt) to indicate different backbone
python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${img_h} ${img_w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                    --caffemodel ${MODEL_PATH}/openpose.caffemodel \
                    --prototxt ${MODEL_PATH}/openpose.prototxt \
                    --input_layout NHWC \
                    --means 128 128 128 \
                    --vars 65536 65536 65536 \
                    --dim_range_min 1 3 ${img_h} ${img_w} \
                    --dim_range_max 32 3 ${img_h} ${img_w} \
                    --type64to32_conversion "true" \
                    --conv_scale_fold "true"  
