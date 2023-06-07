#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} ${MMACTION2_MODEL_IMAGE_SIZE} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --dim_range_min 1 ${MMACTION2_MODEL_IMAGE_SIZE} \
                    --dim_range_max 64 ${MMACTION2_MODEL_IMAGE_SIZE} \
                    --onnx ${MODEL_PATH}/${MMACTION2_MODEL_NAME}.onnx \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --computation_preference fast \
                    --batch_size ${batch_size} \
                    --config ${MMACTION2_MODEL_CONFIG_PATH}