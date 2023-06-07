#!/bin/bash
set -e
set -x

magicmind_model=${1}

precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision} \
                    --input_dims ${batch_size} 3 512 512 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
                    --image_dir ${COCO_DATASETS_PATH}/val2017 \
                    --pytorch_pt ${MODEL_PATH}/ctdet_coco_dlav0_1x_traced_${batch_size}bs.pt \
                    --cluster_num 6 \
                    --input_layout NHWC \
                    --computation_preference fast   \
                    --dim_range_min 1 3 512 512 \
                    --dim_range_max 64 3 512 512 \
                    --means 104 114 120 \
                    --vars  5417 4885 5029 \
                    --type64to32_conversion true \
		    --conv_scale_fold true \
		    --weight_quant_granularity per_tensor \
		    --mlu_arch mtp_372 \
                    --compute_determinism true

