#!/bin/bash

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
if [ $SHAPE_MUTABLE == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/centernet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/centernet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi

if [ ! -f $MAGICMIND_MODEL ];
then
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_${BATCH_SIZE}bs.pt \
                                                  --output_model $MAGICMIND_MODEL \
                                                  --image_dir $DATASETS_PATH/val2017 \
                                                  --precision ${PRECISION} \
                                                  --shape_mutable ${SHAPE_MUTABLE} \
                                                  --batch_size ${BATCH_SIZE}
else
    echo "mm_model: $MAGICMIND_MODEL already exists."
fi

