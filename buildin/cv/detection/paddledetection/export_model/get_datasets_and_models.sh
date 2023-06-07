#!/bin/bash
set -e
set -x

cd ${COCO_DATASETS_PATH}

if [ ! -d "val2017" ];then 
    echo "Downloading val2017.zip"
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -o val2017.zip
else 
    echo "val2017 already exists."
fi

if [ ! -d "annotations" ];then
    echo "Downloading annotations_trainval2017.zip"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -o annotations_trainval2017.zip
else 
    echo "annotations_trainval2017 already exists."
fi

mkdir -p ${MODEL_PATH}
cd ${MODEL_PATH}
if [ -f ${PADDLEDETECTION_MODEL_NAME}.pdparams ];then
    echo "${PADDLEDETECTION_MODEL_NAME} already exists"
else
    echo "Downloading PaddleDetection Model"
    wget -P "${MODEL_PATH}/${PADDLEDETECTION_MODEL_NAME}"  -c ${PADDLEDETECTION_MODEL_PRETRAINED_PATH}
fi

cd ${PROJ_ROOT_PATH}/export_model
if [ -d "PaddleDetection" ];then
    echo "PaddleDetection already exists"
else
    git clone https://github.com/PaddlePaddle/PaddleDetection.git
    cd PaddleDetection
    git checkout 00fe2a1c35603b6fb37b73265aecf6282e5e2ad4
fi

# patch
cd ${PROJ_ROOT_PATH}/export_model 
if grep -q "MMRunner" ${PROJ_ROOT_PATH}/export_model/PaddleDetection/ppdet/engine/trainer.py;
then
    echo "mm_backend.patch already be used"
else
    git apply mm_backend.patch
fi
