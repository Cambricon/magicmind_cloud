#bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

python gen_model.py --precision ${precision}  \
                    --dynamic_shape ${dynamic_shape} \
                    --onnx ${MODEL_PATH}/xdeepfm.onnx \
                    --calibration_data_path ${CRITEO_DATASETS_PATH}/slot_test_data_full \
                    --magicmind_model ${magicmind_model} \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 1 \
                    --input_dims ${batch_size} 13 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 1 \
                    --dim_range_max 2048 1 \
                    --dim_range_min 1 13 \
                    --dim_range_max 2048 13 \
                    --type64to32_conversion true \
                    --calibration_algo linear
