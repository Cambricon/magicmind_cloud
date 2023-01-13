#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
MAX_SEQ_LENGTH=$4

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${QUANT_MODE}_${SHAPE_MUTABLE}
fi

if [ -f $MAGICMIND_MODEL ];
then
  echo "magicmind model: $MAGICMIND_MODEL already exist!!!"
else 
  echo "generate Magicmind model begin..."
  python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt \
                                                --output_model $MAGICMIND_MODEL \
                                                --quant_mode ${QUANT_MODE} \
                                                --shape_mutable ${SHAPE_MUTABLE} \
                                                --batch_size ${BATCH_SIZE} \
                                                --max_seq_length ${MAX_SEQ_LENGTH}
fi
