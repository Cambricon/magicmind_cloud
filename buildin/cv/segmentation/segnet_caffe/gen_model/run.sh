#!/bin/bash
PRECISION=$1 
SHAPE_MUTABLE=$2 
BATCH_SIZE=$3
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/segnet_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/segnet_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/segnet_pascal.caffemodel \
                         --prototxt $MODEL_PATH/segnet_pascal.prototxt \
                         --output_model $MAGICMIND_MODEL \
                         --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages \
                         --precision $PRECISION \
                         --shape_mutable $SHAPE_MUTABLE \
                         --batch_size $BATCH_SIZE \
                         --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
