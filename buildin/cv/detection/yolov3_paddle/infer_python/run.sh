#!/bin/bash
PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
IMAGE_NUM=$4


if [ -d "$PROJ_ROOT_PATH/data/output" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output"
    echo "mkdir sucessed!!!"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/mm_model/yolov3_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                --image_dir $COCO_DATASETS_PATH/val2017 \
                --image_num ${IMAGE_NUM} \
                --file_list $UTILS_PATH/coco_file_list_5000.txt \
                --label_path $UTILS_PATH/coco.names \
                --output_dir $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
