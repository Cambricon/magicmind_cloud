#!/bin/bash
set -e
set -x

QUANT_MODE=$1  
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
BATCH=$4
IMAGE_NUM=$5
if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ]; 
then
    mkdir -p "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
    echo "mkdir sucessed!!!"
else
    echo "output dir exits!!! no need to mkdir again!!!"
fi
bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $PROJ_ROOT_PATH/data/models/retinaface_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                  --image_dir $DATASETS_PATH/WIDER_val/images \
                                  --image_num ${IMAGE_NUM} \
                                  --file_list $PROJ_ROOT_PATH/infer_cpp/wider_val.txt \
                                  --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} \
                                  --save_img true \
                                  --batch ${BATCH}