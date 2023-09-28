#!/bin/bash
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
DEVICE_ID=0

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then
  mkdir "$PROJ_ROOT_PATH/data/output"
  echo "mkdir sucessed!!!"
else
  echo "output dir exits!!! no need to mkdir again!!!"
fi

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_v7_1_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_v7_1_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi
OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
if [ ! -d "$OUTPUT_DIR" ];
then
  mkdir "$OUTPUT_DIR"
fi
cd ../export_model/yolov5/ 
echo "infer Magicmind model..."
python val.py --device_id ${DEVICE_ID} \
	      --magicmind_model ${MAGICMIND_MODEL} \
	      --batch-size ${BATCH_SIZE} \
	      --data ./data/coco.yaml \
	      --img 640 \
	      --conf 0.001 \
	      --iou 0.65 \
