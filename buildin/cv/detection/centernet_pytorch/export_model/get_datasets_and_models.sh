#!/bin/bash

set -e

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
if [ -f "ctdet_coco_dlav0_1x.pth" ];
then
  echo "ctdet_coco_dlav0_1x.pth already exists."
else
  echo "Downloading ctdet_coco_dlav0_1x.pth file"
  pip install gdown
  gdown -c https://drive.google.com/uc?id=18yBxWOlhTo32_swSug_HM4q3BeWgxp_N -O ctdet_coco_dlav0_1x.pth
fi

