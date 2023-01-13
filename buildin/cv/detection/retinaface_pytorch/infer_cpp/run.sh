#!/bin/bash
set -e
set -x

PRECISION=$1  
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
IMAGE_NUM=$4
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
if [ -d $OUTPUT_DIR ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir $OUTPUT_DIR
    echo "mkdir successed!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/retinaface_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_1
else
    MAGICMIND_MODEL=$MODEL_PATH/retinaface_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi


bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $MAGICMIND_MODEL \
                                  --image_dir $DATASETS_PATH/WIDER_val/images \
                                  --image_num $IMAGE_NUM \
                                  --file_list $PROJ_ROOT_PATH/infer_cpp/wider_val.txt \
                                  --output_dir $OUTPUT_DIR \
                                  --save_img true \
                                  --batch $BATCH_SIZE