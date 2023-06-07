#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

splits=(${MMPOSE_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${h} ${w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --dim_range_min 1 3 128 128 \
                    --dim_range_max 64 3 2560 2560 \
                    --onnx ${MODEL_PATH}/${MMPOSE_MODEL_NAME}.onnx \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --computation_preference fast \
                    --batch_size ${batch_size} \
                    --config ${MMPOSE_MODEL_CONFIG_PATH}