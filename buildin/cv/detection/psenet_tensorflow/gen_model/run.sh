#!/bin/bash
if [ ! -d $PROJ_ROOT_PATH/data/mm_model ];
then
    mkdir -p $PROJ_ROOT_PATH/data/mm_model
fi

PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3

if [ -f $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION}    --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                                                    --tf_pb  $MODEL_PATH/psenet.pb \
                                                    --datasets_dir $ICDAR_DATASETS_PATH/icdar2015/images/ \
                                                    --mm_model $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi

