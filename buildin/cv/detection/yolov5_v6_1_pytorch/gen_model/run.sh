#!/bin/bash
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
CONF_THRES=$4
IOU_THRES=$5
MAX_DET=$6

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/yolov5_pytorch_model_${PRECISION}_${SHAPE_MUTABLE}
fi
if [ ! -f $MAGICMIND_MODEL ];
then
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $MODEL_PATH/yolov5m_traced.pt \
                                                  --output_model $MAGICMIND_MODEL \
                                                  --image_dir $DATASETS_PATH/val2017 \
                                                  --precision $PRECISION \
                                                  --shape_mutable $SHAPE_MUTABLE \
                                                  --batch_size $BATCH_SIZE \
                                                  --conf_thres $CONF_THRES \
                                                  --iou_thres $IOU_THRES \
                                                  --max_det $MAX_DET
else
    echo "mm_model: $MAGICMIND_MODEL already exists."
fi
