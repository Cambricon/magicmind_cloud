#!/bin/bash
set -e
set -x

if [ ! -d $MODEL_PATH ];
then
    mkdir $MODEL_PATH
fi

cd $MODEL_PATH
if [ -d "AlexNet_cvgj" ];
then
  echo "AlexNet caffemodel and prototxt already exists."
else
  echo "Downloading alexNet caffemodel and prototxt"
  wget -c https://github.com/cvjena/cnn-models/releases/download/v1.0/cnn-models_cvgj.zip
  unzip -o cnn-models_cvgj.zip
fi

cd $ILSVRC2012_DATASETS_PATH
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi
