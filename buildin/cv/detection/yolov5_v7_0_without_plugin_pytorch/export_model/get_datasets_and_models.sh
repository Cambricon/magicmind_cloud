#!/bin/bash
set -e
set -x


cd ${COCO_DATASETS_PATH}

if [ ! -d "val2017" ];
then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
else 
  echo "val2017 already exists."
fi

if [ ! -d "annotations" ];
then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
else 
  echo "annotations_trainval2017 already exists."
fi

cd $MODEL_PATH
if [ -f "yolov5s.pt" ]; 
then
  echo "yolov5s.pt already exists."
else
  echo "Downloading yolov5s.pt file"
  wget -c https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
fi
