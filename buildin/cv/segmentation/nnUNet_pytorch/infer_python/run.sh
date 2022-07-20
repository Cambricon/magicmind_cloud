#!/bin/bash
PARAMETER_ID=$1
QUANT_MODE=$2
SHAPE_MUTABLE=$3
BATCH_SIZE=$4
BATCH=$5
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID} ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID}"
fi
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/models/magicmind_models/nnUNet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID} \
                --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                --output_folder $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID} \
		--ref_folder $nnUNet_raw_data_base/Task02_Heart/labelsTr
