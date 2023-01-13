#!/bin/bash
PRECISION=$1 
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
PROTOTXT=DenseNet_121.prototxt
CAFFEMODEL=DenseNet_121.caffemodel

if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/densenet121_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/densenet121_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi

if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/$CAFFEMODEL \
                         --prototxt $MODEL_PATH/$PROTOTXT \
                         --output_model $MAGICMIND_MODEL \
                         --image_dir $DATASETS_PATH \
                         --precision $PRECISION \
                         --shape_mutable $SHAPE_MUTABLE \
                         --batch_size $BATCH_SIZE \
                         --input_width 224 \
                         --input_height 224 \
                         --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
