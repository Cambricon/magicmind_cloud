#!/bin/bash
set -e
set -x

precision_mode=$1
model_path=${MODEL_PATH}/ocrnet_${precision_mode}.mm

if [ ! -d $PROJ_ROOT_PATH/data/output/ ];then 
    mkdir -p $PROJ_ROOT_PATH/data/output/
fi

python infer.py --magicmind_model ${model_path} \
                --json_file ${PROJ_ROOT_PATH}/data/output/ocrnet_${precision_mode}.json \
                --data_root ${CITYSCAPES_DATASETS_PATH} \
                --device_id 0 \
                --config $PROJ_ROOT_PATH/export_model/mmsegmentation/configs/ocrnet/ocrnet_hr18_512x1024_160k_cityscapes.py