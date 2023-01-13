#!/bin/bash
PRECISION=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
MAX_SEQ_LENGTH=$4

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/roberta_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}_model
else
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/roberta_${PRECISION}_${SHAPE_MUTABLE}_${MAX_SEQ_LENGTH}_model
fi

if [ -f $MAGICMIND_MODEL ];
then
  echo "magicmind model: $MAGICMIND_MODEL"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --onnx_model $PROJ_ROOT_PATH/data/models/roberta_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.onnx \
                                                --output_model $MAGICMIND_MODEL\
                                                --precision ${PRECISION} \
                                                --shape_mutable ${SHAPE_MUTABLE} \
                                                --batch_size ${BATCH_SIZE} \
                                                --max_seq_length ${MAX_SEQ_LENGTH}
fi
