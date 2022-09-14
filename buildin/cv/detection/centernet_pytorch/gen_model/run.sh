#!/bin/bash

QUANT_MODE=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
if [ ! -f $PROJ_ROOT_PATH/data/models/centernet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/ctdet_coco_dlav0_1x_traced_${BATCH_SIZE}bs.pt \
                                                  --output_model $PROJ_ROOT_PATH/data/models/centernet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                                  --image_dir $DATASETS_PATH/val2017 \
                                                  --quant_mode ${QUANT_MODE} \
                                                  --shape_mutable ${SHAPE_MUTABLE} \
                                                  --batch_size ${BATCH_SIZE}
fi
