#!/bin/bash
PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/resnet50_onnx_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/resnet50_onnx_model_${PRECISION}_${SHAPE_MUTABLE}
fi

if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --onnx_model $MODEL_PATH/resnet50.onnx \
                         --output_model $MAGICMIND_MODEL \
			 --batch_size $BATCH_SIZE  \
                         --image_dir $DATASETS_PATH \
                         --label_file $UTILS_PATH/ILSVRC2012_val.txt\
                         --precision ${PRECISION} \
                         --shape_mutable ${SHAPE_MUTABLE} \
                         --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi