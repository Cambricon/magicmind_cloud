#!/bin/bash
set -e
set -x

INFER(){
    QUANT_MODE=$1  
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    BATCH=$4
    SAVE_IMG=$5
    if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ]; 
    then
        mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
        echo "mkdir sucessed!!!"
    else
        echo "output dir exits!!! no need to mkdir again!!!"
    fi

    if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/voc_preds" ];
    then
        mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/voc_preds"
	echo "mkdir sucessed!!!"
    else
        echo "output dir exits!!! no need to mkdir again!!!"
    fi
    $PROJ_ROOT_PATH/infer_cpp/infer --magicmind_model $PROJ_ROOT_PATH/data/models/ssd_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                    --image_dir $DATASETS_PATH/VOCdevkit \
                                    --output_dir $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH} \
                                    --batch ${BATCH} \
                                    --save_img ${SAVE_IMG}
}

bash build.sh
INFER force_float32 true 1 1 1
