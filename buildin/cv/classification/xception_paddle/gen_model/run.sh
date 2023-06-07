#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --onnx ${MODEL_PATH}/xception41.onnx \
                    --magicmind_model ${magicmind_model} \
                    --input_dims ${batch_size} 3 299 299  \
                    --image_dir ${ILSVRC2012_DATASETS_PATH} \
                    --label_file ${UTILS_PATH}/ILSVRC2012_val.txt\
                    --precision ${precision} \
                    --dynamic_shape ${dynamic_shape} \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --dim_range_min 1 3 299 299 \
                    --dim_range_max 64 3 299 299 \
                    --input_layout NHWC \
                    --means 123.675 116.28 103.53 \
                    --vars 3409.976 3262.694 3291.891
