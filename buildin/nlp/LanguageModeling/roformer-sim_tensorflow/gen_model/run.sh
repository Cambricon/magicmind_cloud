#!/bin/bash
PRECISION=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3  # 2
MAX_SEQ_LENGTH=$4

if [ -f $PROJ_ROOT_PATH/data/models/roformer-sim_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} ];
then
  echo "magicmind model: $PROJ_ROOT_PATH/data/models/roformer-sim_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --pb_model $PROJ_ROOT_PATH/data/models/sim_finish.pb \
                                                --output_model $PROJ_ROOT_PATH/data/models/roformer-sim_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                                                --precision ${PRECISION} \
                                                --shape_mutable ${SHAPE_MUTABLE} \
                                                --batch_size ${BATCH_SIZE} \
                                                --max_seq_length ${MAX_SEQ_LENGTH}
fi