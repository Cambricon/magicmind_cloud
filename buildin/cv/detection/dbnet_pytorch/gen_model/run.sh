#!/bin/bash
PRECISION=$1
SHAPE_MUTABLE=$2
N=$3
H=$4
W=$5
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/dbnet_pt_model_${PRECISION}_${SHAPE_MUTABLE}_${N}_${H}_${W}
else 
    MAGICMIND_MODEL=$MODEL_PATH/dbnet_pt_model_${PRECISION}_${SHAPE_MUTABLE}
fi
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py     --pt_model $MODEL_PATH/dbnet.pt \
                            --dataset_dir $DATASETS_PATH/total_text/test_images \
                            --output_model $MAGICMIND_MODEL \
                            --precision ${PRECISION} \
                            --shape_mutable ${SHAPE_MUTABLE} \
                            --input_height ${H} \
                            --input_width ${W} \
                            --batch_size ${N}
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
