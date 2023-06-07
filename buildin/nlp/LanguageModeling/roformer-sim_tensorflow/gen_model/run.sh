#!/bin/bash
PRECISION=${2} #force_float32/force_float16
SHAPE_MUTABLE=${4} #true/false
BATCH_SIZE=${3}  # 2
MAX_SEQ_LENGTH=${5:-8}
MAGICMIND_MODEL=${1}

if [ $# -lt 4 -o $# -gt 5 ];
then
    echo "bash run.sh <magicmind_model> <precision> <batch_size> <dynamic_shape> <max_seq_length>"
    echo "Usage: bash run.sh ${MODEL_PATH}/roformer_force_float32_false_2_8 force_float32 2 false 8 "
    exit -1
fi

if [ -f ${MAGICMIND_MODEL} ];
then
  echo "magicmind model: ${MAGICMIND_MODEL} already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --tf_pb ${MODEL_PATH}/sim_finish.pb \
                                                --input_names Input-Segment Input-Token \
                                                --output_names Pooler-Dense/BiasAdd \
                                                --magicmind_model ${MAGICMIND_MODEL} \
                                                --precision ${PRECISION} \
                                                --dynamic_shape ${SHAPE_MUTABLE} \
                                                --type64to32_conversion true \
                                                --conv_scale_fold true \
                                                --input_dims ${BATCH_SIZE} ${MAX_SEQ_LENGTH} \
                                                --input_dims ${BATCH_SIZE} ${MAX_SEQ_LENGTH} \
                                                --dim_range_min 1 8 \
                                                --dim_range_min 1 8 \
                                                --dim_range_max 64 ${MAX_SEQ_LENGTH} \
                                                --dim_range_max 64 ${MAX_SEQ_LENGTH} 

fi