#bin/bash
set -e
set -x

# get model
bash get_datasets_and_models.sh
# convert model
cd ${PROJ_ROOT_PATH}/export_model
splits=(${MMSEGMENTATION_MODEL_PRETRAINED_PATH//// })
pth_file=${splits[-1]}

splits=(${MMSEGMENTATION_MODEL_IMAGE_SIZE//,/ })
h=${splits[0]}
w=${splits[1]}

echo ${MMSEGMENTATION_MODEL_CONFIG_PATH}
if [ ! -f ${MODEL_PATH}/${MMSEGMENTATION_MODEL_NAME}.onnx ];then
    python mmsegmentation/tools/pytorch2onnx.py ${MMSEGMENTATION_MODEL_CONFIG_PATH}  \
                                                --checkpoint ${MODEL_PATH}/${pth_file} \
                                                --output-file $MODEL_PATH/${MMSEGMENTATION_MODEL_NAME}.onnx
else
    echo "onnx exist!"
fi
