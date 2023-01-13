#!/bin/bash
set -e
set -x

PRECISION=$1  
SHAPE_MUTABLE=$2
SAVE_IMG=$3
IMAGE_NUM=$4
# fi
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_1
if [ -d $OUTPUT_DIR ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir $OUTPUT_DIR
    echo "mkdir successed!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/centernet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_1
else
    MAGICMIND_MODEL=$MODEL_PATH/centernet_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi


bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model  $MAGICMIND_MODEL \
                                  --image_dir $DATASETS_PATH/val2017 \
                                  --image_num ${IMAGE_NUM} \
                                  --file_list $UTILS_PATH/coco_file_list_5000.txt \
                                  --label_path $UTILS_PATH/coco.names \
                                  --max_bbox_num 100 \
                                  --confidence_thresholds 0.001 \
                                  --output_dir $OUTPUT_DIR \
                                  --save_img true \

