#!/bin/bash
set -e
set -x


if [ -d $PMODEL_PATH ];
then
    echo "folder $PMODEL_PATH already exists"
else
    mkdir $MODEL_PATH
fi

cd $MODEL_PATH
if [ -f "resnet50-v1-7.onnx" ];
then
  echo "resnet50-v1-7.onnx already exists."
else
  echo "Downloading resnet50-v1-7.onnx file"
  wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx
fi

cd $DATASETS_PATH
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi
