#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

splits=(${MMSEGMENTATION_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}
python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 ${h} ${w} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --dim_range_min 1 3 ${h} ${w} \
                    --dim_range_max 16 3 ${h} ${w} \
                    --onnx ${MODEL_PATH}/${MMSEGMENTATION_MODEL_NAME}.onnx \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --computation_preference fast \
                    --image_dir ${CITYSCAPES_DATASETS_PATH}/leftImg8bit/val/lindau \
                    --batch_size ${batch_size} \
