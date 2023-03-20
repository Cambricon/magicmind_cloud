#!/bin/bash
PRECISION=$1 
SHAPE_MUTABLE=$2 
BATCH_SIZE=$3
if [ ${SHAPE_MUTABLE} == 'false' ];
then
    MAGICMIND_MODEL=$MODEL_PATH/volo_d2_model_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
else
    MAGICMIND_MODEL=$MODEL_PATH/volo_d2_model_${PRECISION}_${SHAPE_MUTABLE}
fi

if [ ! -f $MAGICMIND_MODEL ];
then
python gen_model.py --pt_model $MODEL_PATH/volo_d2.pt \
                    --output_model $MAGICMIND_MODEL \
                    --image_dir $DATASETS_PATH \
                    --label_file $UTILS_PATH/ILSVRC2012_val.txt \
                    --precision $PRECISION \
                    --shape_mutable $SHAPE_MUTABLE \
                    --batch_size $BATCH_SIZE \
                    --input_width 224 \
                    --input_height 224 \
                    --device_id 0
else
    echo "mm_model: $MAGICMIND_MODEL already exist."
fi
