#!/bin/bash
PARAMETER_ID=$1
PRECISION=$2
SHAPE_MUTABLE=$3
BATCH_SIZE=$4

if [ ! -d $MODEL_PATH/magicmind_models ];
then
    mkdir $MODEL_PATH/magicmind_models
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID}
else
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${PARAMETER_ID}
fi

if [ ! -f $MAGICMIND_MODEL ];
then
    echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $MODEL_PATH/saved_pts/2dunet_${PARAMETER_ID}.pt \
                                                  --output_model $MAGICMIND_MODEL \
                                                  --precision $PRECISION \
                                                  --shape_mutable $SHAPE_MUTABLE \
                                                  --calib_data_path $MODEL_PATH/saved_pts/calib_data_${PARAMETER_ID}.pt \
                                                  --batch_size $BATCH_SIZE
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
