#!/bin/bash
set -e

PARAMETER_ID=$1
QUANT_MODE=$2
SHAPE_MUTABLE=$3
BATCH_SIZE=$4
BATCH=$5
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID} ];
then
    mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID}"
fi
python save_precessed_data.py --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                              --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                              --output_folder $PROJ_ROOT_PATH/data
bash build.sh
./infer --magicmind_model $PROJ_ROOT_PATH/data/models/magicmind_models/nnUNet_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE}bs_${PARAMETER_ID} \
        --image_dir $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
        --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID} \
	--batch ${BATCH}

python evaluate.py --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                   --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                   --ref_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/labelsTr \
                   --softmax_output_folder $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}bs_${PARAMETER_ID} \
