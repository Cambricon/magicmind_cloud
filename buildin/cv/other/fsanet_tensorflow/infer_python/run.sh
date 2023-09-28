#!/bin/bash
PRECISION=$1 #force_float32
SHAPE_MUTABLE=$2 #true/false
IMAGE_NUM=$3  # max 1969

if [ ! -d "$PROJ_ROOT_PATH/data/output" ];
then 
    mkdir "$PROJ_ROOT_PATH/data/output"
fi

if [ -d "$PROJ_ROOT_PATH/data/output/fsanet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1" ];
then
    echo "output dir already exits!!! no need to mkdir again!!!"
else
    mkdir "$PROJ_ROOT_PATH/data/output/fsanet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1"
    echo "mkdir sucessed!!!"
fi
echo "infer Magicmind model..."
python infer.py --magicmind_model_1 $PROJ_ROOT_PATH/data/mm_model/fsanet_capsule_${PRECISION}_${SHAPE_MUTABLE}_1 \
                --magicmind_model_2 $PROJ_ROOT_PATH/data/mm_model/fsanet_nos_capsule_${PRECISION}_${SHAPE_MUTABLE}_1 \
                --magicmind_model_3 $PROJ_ROOT_PATH/data/mm_model/fsanet_var_capsule_${PRECISION}_${SHAPE_MUTABLE}_1 \
                --image_dir $AFLW2000_DATASETS_PATH/AFLW2000.npz \
                --image_num ${IMAGE_NUM} \
                2>& 1 | tee $PROJ_ROOT_PATH/data/output/fsanet_tf_output_${PRECISION}_${SHAPE_MUTABLE}_1_eval_log 
