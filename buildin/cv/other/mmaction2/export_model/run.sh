#bin/bash
set -e
set -x

# get model
bash get_datasets_and_models.sh
# convert model
cd ${PROJ_ROOT_PATH}/export_model
splits=(${MMACTION2_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}

echo ${MMACTION2_MODEL_CONFIG_PATH}
if [ ! -f ${MODEL_PATH}/${MMACTION2_MODEL_NAME}.onnx ];then
    python mmaction2/tools/deployment/pytorch2onnx.py   ${MMACTION2_MODEL_CONFIG_PATH}  \
                                                        ${MODEL_PATH}/${pth_file} \
                                                        --shape 1 ${MMACTION2_MODEL_IMAGE_SIZE} \
                                                        --show \
                                                        --opset-version 11 \
                                                        --output-file $MODEL_PATH/${MMACTION2_MODEL_NAME}.onnx
else
    echo "onnx exist!"
fi
