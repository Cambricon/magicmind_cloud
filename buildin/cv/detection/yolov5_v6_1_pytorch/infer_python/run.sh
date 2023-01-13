#!/bin/bash
PRECISION=$1
SHAPE_MUTABLE=$2
IMAGE_NUM=$3

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
  echo "mkdir sucessed!!!"
else
  echo "output dir exits!!! no need to mkdir again!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_1
else
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_1
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi
echo "infer Magicmind model..."
python infer.py --magicmind_model $MAGICMIND_MODEL \
                --image_dir $DATASETS_PATH/val2017 \
                --image_num $IMAGE_NUM \
                --file_list $UTILS_PATH/coco_file_list_5000.txt \
                --label_path $UTILS_PATH/coco.names \
                --output_dir $OUTPUT_DIR \
                --save_img true
