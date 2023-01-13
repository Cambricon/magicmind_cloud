#!/bin/bash
PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}"
  echo "mkdir sucessed!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/mm_model/clip_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$PROJ_ROOT_PATH/data/mm_model/clip_onnx_model_${PRECISION}_${SHAPE_MUTABLE}
fi

python infer.py  --device_id 0 \
                 --magicmind_model $MAGICMIND_MODEL \
                 --image_dir $DATASETS_PATH \
                 --result_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/infer_result.txt \
                 --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_labels.txt \
                 --result_top1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_1.txt \
                 --result_top5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}/eval_result_5.txt \
                 --batch_size ${BATCH_SIZE}
