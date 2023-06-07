#!/bin/bash
set -e
set -x

magicmind_det_model=${1}
magicmind_rec_model=${2}
magicmind_cls_model=${3}
precision=${4}
batch_size=${5}
dynamic_shape=${6}

python gen_model.py --task det \
                    --precision ${precision} \
                    --input_dims ${batch_size} 3 704 1280 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_det_model} \
                    --dim_range_min 1 3 32 32 \
                    --dim_range_max 64 3 2560 2560 \
                    --image_dir $ICDAR2015_DATASETS_PATH/det/images \
                    --onnx ${MODEL_PATH}/${PADDLEOCR_DET_MODEL_NAME}.onnx \
                    --mlu_arch mtp_372 \
                    --type64to32_conversion true \
                    --conv_scale_fold true 

python gen_model.py --task rec \
                    --precision ${precision} \
                    --input_dims ${batch_size} 3 48 320 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_rec_model} \
                    --dim_range_min 1 3 32 16 \
                    --dim_range_max 64 3 48 2560 \
                    --image_dir $ICDAR2015_DATASETS_PATH/rec/images \
                    --onnx ${MODEL_PATH}/${PADDLEOCR_REC_MODEL_NAME}.onnx \
                    --mlu_arch mtp_372 \
                    --type64to32_conversion true \
                    --conv_scale_fold true 

python gen_model.py --task cls \
                    --precision ${precision} \
                    --input_dims ${batch_size} 3 48 320 \
                    --dynamic_shape ${dynamic_shape} \
                    --magicmind_model ${magicmind_cls_model} \
                    --dim_range_min 1 3 32 16 \
                    --dim_range_max 64 3 48 2560 \
                    --image_dir $ICDAR2015_DATASETS_PATH/rec/images \
                    --onnx ${MODEL_PATH}/${PADDLEOCR_CLS_MODEL_NAME}.onnx \
                    --mlu_arch mtp_372 \
                    --type64to32_conversion true \
                    --conv_scale_fold true 

