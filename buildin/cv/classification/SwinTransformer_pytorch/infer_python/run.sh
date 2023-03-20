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
    MAGICMIND_MODEL=$MODEL_PATH/swin_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_1
else
    MAGICMIND_MODEL=$MODEL_PATH/swin_onnx_model_${PRECISION}_${SHAPE_MUTABLE}
fi
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_1
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi

echo "before infer"
python infer.py  --device_id 0 \
                 --magicmind_model $MAGICMIND_MODEL \
                 --image_dir $DATASETS_PATH \
                 --image_num $IMAGE_NUM \
                 --name_file $UTILS_PATH/imagenet_name.txt \
                 --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                 --result_file $OUTPUT_DIR/infer_result.txt \
                 --result_label_file $OUTPUT_DIR/eval_labels.txt \
                 --result_top1_file $OUTPUT_DIR/eval_result_1.txt \
                 --result_top5_file $OUTPUT_DIR/eval_result_5.txt 
echo "infer success !!!"