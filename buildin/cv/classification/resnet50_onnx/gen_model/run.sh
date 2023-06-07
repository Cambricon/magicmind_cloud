#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 224 224 \
                    --batch_size ${batch_size} \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${ILSVRC2012_DATASETS_PATH} \
                    --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                    --input_layout NHWC \
                    --dim_range_min 1 3 224 224 \
                    --dim_range_max 64 3 224 224 \
                    --means 123.675 116.28 103.53 \
                    --vars  3409.976 3262.694 3291.891 \
                    --onnx $MODEL_PATH/resnet50-v1-7.onnx






