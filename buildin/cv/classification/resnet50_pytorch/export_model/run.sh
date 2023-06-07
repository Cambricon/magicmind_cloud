#!/bin/bash
set -e
set -x

# get_datasets_and_models
bash get_datasets_and_models.sh

cd ${MODEL_PATH}
if [ -f "resnet50.pt" ];
then
    echo "resnet50.pt already exists"
else
    echo "begin generate pt model"
    cd ${PROJ_ROOT_PATH}/export_model
    python export.py
    echo "generate pt model success"
fi
