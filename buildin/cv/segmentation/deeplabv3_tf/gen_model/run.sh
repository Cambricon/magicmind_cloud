#!/bin/bash
QUANT_MODE=$1 
SHAPE_MUTABLE=$2 
if [ ! -f $MODEL_PATH/deeplabv3_tf_model_${QUANT_MODE}_${SHAPE_MUTABLE} ];
then
    python gen_model.py  --tf_model $MODEL_PATH/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb \
                         --output_model_path $MODEL_PATH/deeplabv3_tf_model_${QUANT_MODE}_${SHAPE_MUTABLE} \
                         --image_dir $DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages \
                         --file_list $DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                         --quant_mode ${QUANT_MODE} \
                         --shape_mutable ${SHAPE_MUTABLE}
else
    echo "mm_model: $MODEL_PATH/deeplabv3_tf_model_${QUANT_MODE}_${SHAPE_MUTABLE} already exist."
fi
