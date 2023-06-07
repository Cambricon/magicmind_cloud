#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 1 32 200 \
		    --type64to32_conversion true \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --pytorch_pt ${MODEL_PATH}/crnn.pt \
                    --dim_range_min 1 1 32 100 \
                    --dim_range_max 32 1 32 300 \


