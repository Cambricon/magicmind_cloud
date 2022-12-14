#!/bin/bash
QUANT_MODE=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
BATCH_SIZE=$3
BATCH=$4
IMAGE_NUM=$5
if [ -d "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --magicmind_model $MODEL_PATH/yolov5_pytorch_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                --image_dir $DATASETS_PATH/val2017 \
                --image_num ${IMAGE_NUM} \
                --file_list $UTILS_PATH/coco_file_list_5000.txt \
                --label_path $UTILS_PATH/coco.names \
                --batch ${BATCH} \
                --output_dir $PROJ_ROOT_PATH/data/output/infer_python_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} \
                --save_img true
