#!/bin/bash
PARAMETER_ID=$1
QUANT_MODE=$2
SHAPE_MUTABLE=$3
BATCH_SIZE=$4

if [ ! -d $PROJ_ROOT_PATH/data/models/magicmind_models ];
then
    mkdir $PROJ_ROOT_PATH/data/models/magicmind_models
fi
if [ ! -f $PROJ_ROOT_PATH/data/models/magicmind_models/nnUNet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID} ];
then
    echo "generate Magicmind model begin..."
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $PROJ_ROOT_PATH/data/models/saved_pts/${BATCH_SIZE}bs/2dunet_${PARAMETER_ID}.pt \
                                                  --output_model $PROJ_ROOT_PATH/data/models/magicmind_models/nnUNet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID} \
                                                  --quant_mode ${QUANT_MODE} \
                                                  --shape_mutable ${SHAPE_MUTABLE} \
                                                  --calib_data_path $PROJ_ROOT_PATH/data/models/saved_pts/${BATCH_SIZE}bs/calib_data_${PARAMETER_ID}.pt \
						  --batch_size ${BATCH_SIZE}
fi
