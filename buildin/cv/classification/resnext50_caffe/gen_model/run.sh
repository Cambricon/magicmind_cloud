#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
PROTOTXT=deploy_resnext50-32x4d.prototxt
CAFFEMODEL=resnext50-32x4d.caffemodel
if [ ! -f $MODEL_PATH/${MODEL_NAME}_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/$CAFFEMODEL \
                         --prototxt $MODEL_PATH/$PROTOTXT \
                         --output_model $PROJ_ROOT_PATH/data/models/${MODEL_NAME}_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                         --image_dir $DATASETS_PATH \
                         --quant_mode ${QUANT_MODE} \
                         --shape_mutable ${SHAPE_MUTABLE} \
                         --batch_size ${BATCH_SIZE} \
                         --input_width 224 \
                         --input_height 224 \
                         --device_id 0
else
    echo "mm_model: $MODEL_PATH/${MODEL_NAME}_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist."
fi
