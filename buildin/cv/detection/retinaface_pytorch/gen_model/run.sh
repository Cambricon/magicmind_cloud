#!/bin/bash
set -e 
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --pytorch_pt ${PROJ_ROOT_PATH}/data/models/retinaface_traced.pt \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${WIDERFACE_DATASETS_PATH}/WIDER_val \
                    --precision ${precision} \
                    --input_dims ${batch_size} 3 672 1024  \
                    --dynamic_shape ${dynamic_shape} \
                    --type64to32_conversion true \
                    --conv_scale_fold true \
                    --dim_range_min 1 3 672 1024 \
                    --dim_range_max 64 3 672 1024 \
                    --weight_quant_granularity per_axis \
                    --means 104 117 123 \
                    --vars 1 1 1 \
                    --input_layout NHWC \
                    --mlu_arch mtp_372 



