#!/bin/bash
set -e
set -x

magicmind_model=${1}
precision=${2}
batch_size=${3}
dynamic_shape=${4}

echo "PaddleDection ${PADDLEDETECTION_MODEL_NAME} Inference Only Support batch_size=1 and dynamic_shape=false"

if [ ${PADDLEDETECTION_MODEL_NAME} = "yolov3_darknet53_270e_coco" ] ;then
    python gen_model.py --precision ${precision} \
                        --input_dims ${batch_size} 2 \
                        --input_dims ${batch_size} 3 ${PADDLEDETECTION_MODEL_INPUT_SIZE} ${PADDLEDETECTION_MODEL_INPUT_SIZE} \
                        --input_dims ${batch_size} 2 \
                        --dynamic_shape ${dynamic_shape} \
                        --magicmind_model ${magicmind_model} \
                        --image_dir ${COCO_DATASETS_PATH}/val2017 \
                        --onnx ${MODEL_PATH}/${PADDLEDETECTION_MODEL_NAME}.onnx \
                        --model_name yolov3 \
                        --type64to32_conversion true \
                        --conv_scale_fold true
elif [ ${PADDLEDETECTION_MODEL_NAME} = "ppyoloe_crn_s_400e_coco" ] ;then
    python gen_model.py --precision ${precision} \
                        --input_dims ${batch_size} 3 ${PADDLEDETECTION_MODEL_INPUT_SIZE} ${PADDLEDETECTION_MODEL_INPUT_SIZE} \
                        --input_dims ${batch_size} 2 \
                        --dynamic_shape ${dynamic_shape} \
                        --magicmind_model ${magicmind_model} \
                        --image_dir ${COCO_DATASETS_PATH}/val2017 \
                        --onnx ${MODEL_PATH}/${PADDLEDETECTION_MODEL_NAME}.onnx \
                        --model_name ppyoloe \
                        --type64to32_conversion true \
                        --conv_scale_fold true 
else 
    echo "Not surpport ${PADDLEDETECTION_MODEL_NAME} !"
fi

