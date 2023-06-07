#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

splits=(${MMDETECTION_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}
python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${h} ${w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --dim_range_min 1 3 224 224 \
                    --dim_range_max 64 3 1344 1920 \
                    --onnx ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}.onnx \
                    --type64to32_conversion true \
		    --cluster_num 6 \
                    --conv_scale_fold true \
                    --config ${MMDETECTION_MODEL_CONFIG_PATH} \
                    --batch_size ${batch_size}
