#!/bin/bash
PRECISION=$1 #force_float32/force_float16/qint8_mixed_float16
SHAPE_MUTABLE=$2 #true/false
IMAGE_NUM=$3

if [ ! -f pse/pse.so ];
then
    pushd pse
    make
    popd
fi
if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then 
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/psenet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/psenet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1"
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --magicmind_model $PROJ_ROOT_PATH/data/mm_model/psenet_tf_${PRECISION}_${SHAPE_MUTABLE}_1 \
                --image_dir $ICDAR_DATASETS_PATH/icdar2015/images/ \
                --image_num ${IMAGE_NUM} \
                --batch_size 1 \
                --output_dir $PROJ_ROOT_PATH/data/output/psenet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1 \
                --save_img true \
                --save_json true \
                --json_path $PROJ_ROOT_PATH/data/output/psenet_tf_result_${PRECISION}_${SHAPE_MUTABLE}_1.json
