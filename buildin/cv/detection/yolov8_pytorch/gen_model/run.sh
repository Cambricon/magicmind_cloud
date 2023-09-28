#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

if [ ! -f $magicmind_model ];
then
    python ${PROJ_ROOT_PATH}/gen_model/gen_model.py --onnx ${MODEL_PATH}/yolov8n.onnx \
                                                  --magicmind_model ${magicmind_model} \
                                                  --input_dims ${batch_size} 3 640 640 \
                                                  --image_dir ${COCO_DATASETS_PATH}/val2017 \
                                                  --precision ${precision} \
                                                  --dynamic_shape ${dynamic_shape} \
                                                  --dim_range_min 1 3 128 128 \
                                                  --dim_range_max 64 3 960 960 \
                                                  --type64to32_conversion true \
                                                  --activation_quant_algo symmetric \
                                                  --weight_quant_granularity per_axis \
                                                  --batch_size ${batch_size} 
else
    echo "mm_model: ${magicmind_model} already exists."
fi
