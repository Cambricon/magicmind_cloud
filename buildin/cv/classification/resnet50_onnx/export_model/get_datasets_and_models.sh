#!/bin/bash
set -e
set -x

if [ -d $PROJ_ROOT_PATH/data ];
then 
    echo "folder $PROJ_ROOT_PATH/data already exists"
else
    mkdir $PROJ_ROOT_PATH/data
fi

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exists"
else
    mkdir $PROJ_ROOT_PATH/data/models
fi

cd $MODEL_PATH
if [ -f "resnet50-v1-7.onnx" ];
then
  echo "resnet50-v1-7.onnx already exists."
else
  echo "Downloading resnet50-v1-7.onnx file"
  wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx
fi
