#!/bin/bash
set -e
magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}
max_seq_length=384

python gen_model.py --precision ${precision} \
                    --pt_input_dtypes INT32 INT32 INT32 \
                    --input_dtypes INT32 INT32 INT32 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_model} \
		    --type64to32_conversion true \
		    --cluster_num 6 8 \
                    --pytorch_pt ${MODEL_PATH}/bert_squad_pytorch_384.pt \
                    --input_dims ${batch_size} ${max_seq_length} \
                    --input_dims ${batch_size} ${max_seq_length} \
                    --input_dims ${batch_size} ${max_seq_length} \
                    --dim_range_min 1 ${max_seq_length} \
                    --dim_range_min 1 ${max_seq_length} \
                    --dim_range_min 1 ${max_seq_length} \
                    --dim_range_max 64 ${max_seq_length} \
                    --dim_range_max 64 ${max_seq_length} \
                    --dim_range_max 64 ${max_seq_length} 
    
