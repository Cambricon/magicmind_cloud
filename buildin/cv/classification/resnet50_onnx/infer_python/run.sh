#!/bin/bash
QUANT_MODE=$1 #forced_float32/forced_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
BATCH=$4
IMAGE_NUM=$5
if [ -d "$PROJ_ROOT_PATH/data/output" ];
then
  echo "folder:$PROJ_ROOT_PATH/data/output already exits!!! no need to mkdir again!!!"
else
  mkdir "$PROJ_ROOT_PATH/data/output"
  echo "mkdir sucessed!!!"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ];
then
  echo "folder:$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} already exits!!! no need to mkdir again!!!"
else
  mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
  echo "mkdir sucessed!!!"
fi
python infer.py  --device_id 0 \
                 --magicmind_model $PROJ_ROOT_PATH/data/models/resnet50_onnx_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                 --image_dir $DATASETS_PATH/images \
                 --image_num ${IMAGE_NUM} \
                 --name_file $DATASETS_PATH/names.txt \
                 --label_file $DATASETS_PATH/imagenet_1000.txt \
                 --result_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/infer_result.txt \
                 --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_labels.txt \
                 --result_top1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_1.txt \
                 --result_top5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_5.txt \
                 --batch_size ${BATCH}
