#!/bin/bash
PRECISION=$1 
SHAPE_MUTABLE=$2 
BATCH_SIZE=$3
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/squeezenet_v1_1_caffe_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/squeezenet_v1_1_caffe_model_${PRECISION}_${SHAPE_MUTABLE}
fi
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/squeezenet_v1_1.caffemodel \
                         --prototxt $MODEL_PATH/deploy_v1_1.prototxt \
                         --output_model $MAGICMIND_MODEL \
                         --image_dir $DATASETS_PATH \
                         --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                         --precision $PRECISION \
                         --shape_mutable $SHAPE_MUTABLE \
                         --batch_size $BATCH_SIZE \
                         --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
