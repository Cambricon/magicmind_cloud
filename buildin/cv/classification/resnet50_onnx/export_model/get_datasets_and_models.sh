#!/bin/bash
set -e
set -x


if [ -d $MODEL_PATH ];
then
    echo "folder $MODEL_PATH already exists"
else
    mkdir -p $MODEL_PATH
fi

cd $MODEL_PATH
if [ -f "resnet50-v1-7.onnx" ];
then
  echo "resnet50-v1-7.onnx already exists."
else
  echo "Downloading resnet50-v1-7.onnx file"
  wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx
fi

cd $ILSVRC2012_DATASETS_PATH
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi
