#!/bin/bash
set -e
set -x

if [ -d ${MODEL_PATH} ];
then
    echo "folder ${MODEL_PATH} already exist!!!"
else
    mkdir -p "${MODEL_PATH}"
fi

# 1. prepare data and models
bash get_datasets_and_models.sh

# 2. git clone source code
if [ -d "tensorflow-yolov3" ];
then
    echo "tensorflow-yolov3 already exist."
else
    echo "git clone tensorflow-yolov3"
    git clone https://github.com/YunYang1994/tensorflow-yolov3
    cd tensorflow-yolov3
    git reset --hard 03cb272af2e26d598c553f3a2d38024fc6f67a0b
fi

# 3. patch
if grep -q "../../"  ${PROJ_ROOT_PATH}/export_model/tensorflow-yolov3/convert_weight.py;
then
    echo "patch already be used"
else
    cd ${PROJ_ROOT_PATH}/export_model 
    patch -p0 tensorflow-yolov3/convert_weight.py < convert.patch
    patch -p0 tensorflow-yolov3/freeze_graph.py < freeze.patch
fi

# 4. export pb 
cd ${PROJ_ROOT_PATH}/export_model/tensorflow-yolov3
python convert_weight.py
python freeze_graph.py

