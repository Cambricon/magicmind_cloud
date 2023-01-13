#!/bin/bash
PARAMETER_ID=$1
PRECISION=$2
SHAPE_MUTABLE=$3
if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_1bs_${PARAMETER_ID}
else
    MAGICMIND_MODEL=$MODEL_PATH/magicmind_models/nnUNet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${PARAMETER_ID}
fi

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_1bs_${PARAMETER_ID}
if [ ! -d $OUTPUT_DIR ];
then
    mkdir $OUTPUT_DIR
fi
python infer.py --magicmind_model $MAGICMIND_MODEL \
                --model_path $MODEL_PATH/2d/Task002_Heart/nnUNetTrainerV2__nnUNetPlansv2.1 \
                --data_folder $nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/imagesTr \
                --output_folder $OUTPUT_DIR \
                --ref_folder $nnUNet_raw_data_base/Task02_Heart/labelsTr 
