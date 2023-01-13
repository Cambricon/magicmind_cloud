#!/bin/bash
set -e

PARAMETER_ID=$1
PRECISION=$2
SHAPE_MUTABLE=$3
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir $PROJ_ROOT_PATH/data/output
fi
if [ $SHAPE_MUTABLE == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_1bs_${PARAMETER_ID}
else
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${PARAMETER_ID}
fi

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1bs_${PARAMETER_ID}
if [ ! -d $OUTPUT_DIR ];
then
    mkdir $OUTPUT_DIR
fi

python save_precessed_data.py --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                              --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                              --output_folder $OUTPUT_DIR
                              
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $MAGICMIND_MODEL \
                                --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                                --output_folder $OUTPUT_DIR 

python evaluate.py --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                   --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                   --ref_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/labelsTr \
                   --softmax_output_folder $OUTPUT_DIR
