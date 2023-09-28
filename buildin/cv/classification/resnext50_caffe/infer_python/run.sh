#!/bin/bash
set -x
set -e
PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
IMAGE_NUM=$4
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir -p "$PROJ_ROOT_PATH/data/output"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}" ];
then
  echo "output dir already exits!!! no need to mkdir again!!!"
else
  mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
  echo "mkdir sucessed!!!"
fi
python infer.py  --device_id 0 \
                 --magicmind_model $MODEL_PATH/${MODEL_NAME}_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                 --image_dir $ILSVRC2012_DATASETS_PATH \
                 --image_num ${IMAGE_NUM} \
                 --name_file $UTILS_PATH/imagenet_name.txt \
                 --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                 --result_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/infer_result.txt \
                 --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_labels.txt \
                 --result_top1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_1.txt \
                 --result_top5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_5.txt \
                 --batch ${BATCH_SIZE}
