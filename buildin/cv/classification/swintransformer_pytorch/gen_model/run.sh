#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 224 224 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${ILSVRC2012_DATASETS_PATH} \
                    --onnx ${MODEL_PATH}/swin.onnx \
                    --input_layout NHWC \
		    --type64to32_conversion true \
                    --dim_range_min 1 3 224 224 \
                    --dim_range_max 64 3 224 224 \
                    --means 123.675 116.28 103.53 \
                    --vars 3409.976 3262.694 3291.89 
