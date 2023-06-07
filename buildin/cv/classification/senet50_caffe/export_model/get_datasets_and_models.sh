#!/bin/bash
set -e
set -x

if [ -d ${MODEL_PATH} ];
then
    echo "folder ${MODEL_PATH} already exists"
else
    mkdir -p ${MODEL_PATH}
fi

cd ${MODEL_PATH}
if [ -f "SE-ResNet-50.caffemodel" ];
then
  echo "senet50 caffemodel already exists."
else
  echo "Downloading senet50 caffemodel file on https://github.com/hujie-frank/SENet to SE-ResNet-50.caffemodel"
  exit 1
fi

if [ -f "SE-ResNet-50.prototxt" ];
then
  echo "senet50 prototxt file already exists."
else
  echo "Downloading senet50 prototxt file on https://github.com/hujie-frank/SENet to SE-ResNet-50.prototxt "
  exit 1
fi

cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi
