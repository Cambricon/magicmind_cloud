#!/bin/bash
onnx_path=${MODEL_PATH}/roberta_1bs_128.onnx

MAGICMIND_MODEL=${1}
PRECISION=${2}
BATCH_SIZE=${3}
SHAPE_MUTABLE=${4}
MAX_SEQ_LENGTH=${5:-128}
ONNX_MODEL=${6:-${onnx_path}}

if [ $# -lt 4 -o $# -gt 6 ];
then
    echo "bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape> <max_seq_length> <onnx_path>"
    echo "Usage: bash run.sh ${MODEL_PATH}/roberta_force_float32_false_1_128 force_float32 1 false 128 ${MODEL_PATH}/roberta_1bs_128.onnx "
    exit -1
fi

if [ -f ${MAGICMIND_MODEL} ];
then
  echo "magicmind model: $MAGICMIND_MODEL exist"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --onnx ${ONNX_MODEL} \
                                                --magicmind_model ${MAGICMIND_MODEL} \
                                                --precision ${PRECISION} \
                                                --dynamic_shape ${SHAPE_MUTABLE} \
                                                --type64to32_conversion true \
                                                --conv_scale_fold true \
                                                --input_dims ${BATCH_SIZE} ${MAX_SEQ_LENGTH} \
                                                --input_dims ${BATCH_SIZE} ${MAX_SEQ_LENGTH} \
                                                --input_dims ${BATCH_SIZE} ${MAX_SEQ_LENGTH} \
                                                --dim_range_min 1 ${MAX_SEQ_LENGTH} \
                                                --dim_range_min 1 ${MAX_SEQ_LENGTH} \
                                                --dim_range_min 1 ${MAX_SEQ_LENGTH} \
                                                --dim_range_max 64 ${MAX_SEQ_LENGTH} \
                                                --dim_range_max 64 ${MAX_SEQ_LENGTH} \
                                                --dim_range_max 64 ${MAX_SEQ_LENGTH}
fi
