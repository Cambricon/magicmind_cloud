#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 640 959 \
		            --type64to32_conversion true \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --pytorch_pt ${MODEL_PATH}/unet_carvana_scale0.5_epoch2_trace.pt \
                    --means 0 0 0 \
                    --vars  65025 65025 65025 \
                    --dim_range_min 1 3 640 959 \
                    --dim_range_max 64 3 640 959 \
		    --image_dir ${CARVANA_DATASETS_PATH}/imgs \
		    --device_id 0 \
		    --scale 0.5

