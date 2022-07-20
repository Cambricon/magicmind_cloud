#!/bin/bash
set -e
set -x

INFER(){
    QUANT_MODE=$1  
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    BATCH=$4
    IMAGE_NUM=$5
    if [ ! -d "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}" ]; 
    then
      mkdir "$PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}"
      echo "mkdir sucessed!!!"
    else
      echo "output dir exits!!! no need to mkdir again!!!"
    fi
    $PROJ_ROOT_PATH/infer_cpp/infer   --magicmind_model $PROJ_ROOT_PATH/data/models/vgg16_caffe_model_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH_SIZE} \
                                      --image_dir $DATASETS_PATH/images \
                                      --image_num ${IMAGE_NUM} \
                                      --name_file $DATASETS_PATH/names.txt \
                                      --label_file $DATASETS_PATH/imagenet_1000.txt \
                                      --result_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/infer_result.txt \
                                      --result_label_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_labels.txt \
                                      --result_top1_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_1.txt \
                                      --result_top5_file $PROJ_ROOT_PATH/data/output/infer_cpp_output_${QUANT_MODE}_${SHAPE_MUTABLE}_${BATCH}/eval_result_5.txt \
                                      --batch ${BATCH}
}

bash build.sh
INFER force_float32 true 1 1 1000
