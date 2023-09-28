#!/bin/bash
PRECISION=$1 
MAGICMIND_MODEL=$MODEL_PATH/deeplabv3_tensorflow_model_${PRECISION}
if [ ! -f $MAGICMIND_MODEL ];
then
    python gen_model.py  --tf_model $MODEL_PATH/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb \
                         --output_model_path $MAGICMIND_MODEL \
                         --image_dir $VOC2012_DATASETS_PATH/VOCdevkit/VOC2012/JPEGImages \
                         --file_list $VOC2012_DATASETS_PATH/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
                         --precision $PRECISION
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
