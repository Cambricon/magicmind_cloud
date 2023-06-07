#bin/bash
set -e
set -x

# get model
bash get_datasets_and_models.sh

# convert model
cd ${PROJ_ROOT_PATH}/export_model
splits=(${MMPOSE_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}

splits=(${MMPOSE_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}

echo ${MMPOSE_MODEL_CONFIG_PATH}
if [ ! -f ${MODEL_PATH}/${MMPOSE_MODEL_NAME}.onnx ];then
    python mmpose/tools/deployment/pytorch2onnx.py  ${MMPOSE_MODEL_CONFIG_PATH}  \
                                                    ${MODEL_PATH}/${pth_file} \
                                                    --shape 1 3 ${h} ${w} \
                                                    --opset-version 11 \
                                                    --output-file $MODEL_PATH/${MMPOSE_MODEL_NAME}.onnx
else
    echo "onnx exist!"
fi
