#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 224 224 \
                    --input_dims 100 77 \
                    --batch_size ${batch_size} 100 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${CIFAR100_DATASETS_PATH} \
                    --dim_range_min 1 3 224 224 \
                    --dim_range_max 64 3 224 224 \
                    --dim_range_min 100 77 \
                    --dim_range_max 100 77 \
                    --onnx $MODEL_PATH/clip.onnx \
                    --type64to32_conversion "true" \
                    --conv_scale_fold "true"  \
                    --device_id 0





        
