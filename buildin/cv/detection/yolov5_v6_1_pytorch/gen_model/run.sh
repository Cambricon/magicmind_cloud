#!/bin/bash

QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
CONF_THRES=$4
IOU_THRES=$5
MAX_DET=$6
if [ ! -f $MODEL_PATH/yolov5_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    python $PROJ_ROOT_PATH/gen_model/gen_model.py --pt_model $MODEL_PATH/yolov5m_traced.pt \
                                                  --output_model $MODEL_PATH/yolov5_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                                  --image_dir $DATASETS_PATH/val2017 \
                                                  --quant_mode ${QUANT_MODE} \
                                                  --shape_mutable ${SHAPE_MUTABLE} \
                                                  --batch_size ${BATCH_SIZE} \
                                                  --conf_thres ${CONF_THRES} \
                                                  --iou_thres ${IOU_THRES} \
                                                  --max_det ${MAX_DET}
fi
