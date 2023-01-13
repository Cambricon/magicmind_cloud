#!/bin/bash
if [ ! -d $PROJ_ROOT_PATH/data/mm_model ];
then
    mkdir -p $PROJ_ROOT_PATH/data/mm_model
fi

PRECISION=$1 # only support fp32
SHAPE_MUTABLE=$2 
BATCH_SIZE=$3
# capsule model
if [ -f $PROJ_ROOT_PATH/data/mm_model/fsanet_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/fsanet_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION}    --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                                                    --tf_pb  $MODEL_PATH/fsanet_capsule_3_16_2_21_5.pb \
                                                    --mm_model $PROJ_ROOT_PATH/data/mm_model/fsanet_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/fsanet_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi

# noS_capsule model
if [ -f $PROJ_ROOT_PATH/data/mm_model/fsanet_nos_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/fsanet_nos_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION}    --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                                                    --tf_pb  $MODEL_PATH/fsanet_noS_capsule_3_16_2_192_5.pb \
                                                    --mm_model $PROJ_ROOT_PATH/data/mm_model/fsanet_nos_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/fsanet_nos_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi

# var_capsule model
if [ -f $PROJ_ROOT_PATH/data/mm_model/fsanet_var_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    echo "magicmind model: $PROJ_ROOT_PATH/data/mm_model/fsanet_var_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist!"
else 
    echo "generate Magicmind model begin..."
    cd $PROJ_ROOT_PATH/gen_model/
    python gen_model.py --precision ${PRECISION}    --batch_size $BATCH_SIZE --shape_mutable ${SHAPE_MUTABLE} \
                                                    --tf_pb  $MODEL_PATH/fsanet_var_capsule_3_16_2_21_5.pb \
                                                    --mm_model $PROJ_ROOT_PATH/data/mm_model/fsanet_var_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    echo "Generate model done, model save to $PROJ_ROOT_PATH/data/mm_model/fsanet_var_capsule_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
fi