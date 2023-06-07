#!/bin/bash
set -e
set -x
MAGICMIND_MODEL=${1}
PRECISION=${2} #force_float32/force_float16
BATCH_SIZE=${3}
SHAPE_MUTABLE=${4} #true/false
MAX_SEQ_LENGTH=${5:-384}

if [ $# -lt 4 -o $# -gt 5 ];
then
  echo "bash run.sh <magicmind_model> <precision> <batch_size> <shape_mutable> <max_seq_length> "
  echo "Usage: bash run.sh ${MODEL_PATH}/bert_force_float32_false_1_384 force_float32 1 false 384 "
  exit -1
fi

if [ -f $MAGICMIND_MODEL ];
then
  echo "magicmind model: $MAGICMIND_MODEL already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --tf_pb $MODEL_PATH/frozen_graph.pb \
                                                --input_names IteratorGetNext:0 IteratorGetNext:1 IteratorGetNext:2 \
                                                --output_names unstack:0 unstack:1 \
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
