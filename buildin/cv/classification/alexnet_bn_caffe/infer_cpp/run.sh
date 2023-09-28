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
  echo "mkdir sucessed!!!"
else
  echo "output dir exits!!! no need to mkdir again!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/alexnet_bn_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/alexnet_bn_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_cpp_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi

bash build.sh
$PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $MAGICMIND_MODEL \
                                  --image_dir $ILSVRC2012_DATASETS_PATH \
                                  --image_num $IMAGE_NUM \
                                  --name_file $UTILS_PATH/imagenet_name.txt \
                                  --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                                  --result_file $OUTPUT_DIR/infer_result.txt \
                                  --result_label_file $OUTPUT_DIR/eval_labels.txt \
                                  --result_top1_file $OUTPUT_DIR/eval_result_1.txt \
                                  --result_top5_file $OUTPUT_DIR/eval_result_5.txt \
                                  --batch_size $BATCH_SIZE

