#!/bin/bash
PRECISION=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
MAX_SEQ_LENGTH=$4

if [ -f $MODEL_PATH/bert_tensorflow_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} ];
then
  echo "magicmind model: $MODEL_PATH/bert_tensorflow_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --pb_model $MODEL_PATH/frozen_graph.pb \
                                                --output_model $MODEL_PATH/bert_tensorflow_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}_${MAX_SEQ_LENGTH} \
                                                --precision ${PRECISION} \
                                                --shape_mutable ${SHAPE_MUTABLE} \
                                                --batch_size ${BATCH_SIZE} \
                                                --max_seq_length ${MAX_SEQ_LENGTH}
fi
