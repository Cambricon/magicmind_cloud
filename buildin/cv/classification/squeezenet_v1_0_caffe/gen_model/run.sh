#!/bin/bash
QUANT_MODE=$1 #forced_float32/forced_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
if [ ! -f $PROJ_ROOT_PATH/data/models/squeezenet_v1_0_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} ];
then
    python gen_model.py  --caffe_model $MODEL_PATH/squeezenet_v1_0.caffemodel \
                         --prototxt $MODEL_PATH/deploy_v1_0.prototxt \
                         --output_model $PROJ_ROOT_PATH/data/models/squeezenet_v1_0_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                         --image_dir $DATASETS_PATH/images \
                         --label_file $DATASETS_PATH/imagenet_1000.txt \
                         --quant_mode ${QUANT_MODE} \
                         --shape_mutable ${SHAPE_MUTABLE} \
                         --batch_size ${BATCH_SIZE} \
                         --input_width 224 \
                         --input_height 224 \
                         --device_id 0
else
    echo "mm_model: $PROJ_ROOT_PATH/data/models/squeezenet_v1_0_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} already exist."
fi
