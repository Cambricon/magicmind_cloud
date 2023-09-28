#!/bin/bash
PRECISION=$1
SHAPE_MUTABLE=$2
BATCH_SIZE=$3
if [ $SHAPE_MUTABLE == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/ssd_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/ssd_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/mobilenet_iter_73000.caffemodel \
                         --prototxt $MODEL_PATH/deploy.prototxt \
                         --output_model $MAGICMIND_MODEL \
                         --image_dir $VOC2007_DATASETS_PATH/VOCdevkit/VOC2007/JPEGImages \
                         --precision $PRECISION \
                         --shape_mutable $SHAPE_MUTABLE \
                         --batch_size $BATCH_SIZE \
                         --input_width 300 \
                         --input_height 300 \
                         --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exists."
fi
