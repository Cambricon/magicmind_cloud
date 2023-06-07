#bin/bash
set -e
set -x

# get model
bash get_datasets_and_models.sh
# convert model
cd ${PROJ_ROOT_PATH}/export_model
splits=(${MMDETECTION_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}

splits=(${MMDETECTION_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}

echo ${MMDETECTION_MODEL_CONFIG_PATH}
if [ ! -f ${MODEL_PATH}/${MMDETECTION_MODEL_NAME}.onnx ];then
    python mmdetection/tools/deployment/pytorch2onnx.py --output-file $MODEL_PATH/${MMDETECTION_MODEL_NAME}.onnx \
                                                        ${MMDETECTION_MODEL_CONFIG_PATH}  \
                                                        ${MODEL_PATH}/${pth_file} \
                                                        --shape ${h} ${w} \
                                                        --dynamic-export
else
    echo "onnx exist!"
fi
