#!/bin/bash
PRECISION=$1
MAGICMIND_MODEL=$MODEL_PATH/crnn_pt_model_${PRECISION}
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --pt_model $MODEL_PATH/crnn.pt \
                         --output_model $MAGICMIND_MODEL \
                         --precision ${PRECISION}
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
