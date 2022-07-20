#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
if [ ! -f $PROJ_ROOT_PATH/data/models/ssd_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/mobilenet_iter_73000.caffemodel \
                         --prototxt $MODEL_PATH/deploy.prototxt \
                         --output_model $PROJ_ROOT_PATH/data/models/ssd_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                         --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages \
                         --quant_mode ${QUANT_MODE} \
                         --shape_mutable ${SHAPE_MUTABLE} \
                         --batch_size ${BATCH_SIZE} \
                         --input_width 300 \
                         --input_height 300 \
                         --device_id 0
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/ssd_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist."
fi

