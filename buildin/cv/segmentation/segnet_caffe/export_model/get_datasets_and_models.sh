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
if [ -f "segnet_pascal.caffemodel" ];
then
  echo "segnet caffemodel already exists."
else
  echo "Downloading segnet caffemodel file"
  wget -c http://mi.eng.cam.ac.uk/~agk34/resources/SegNet/segnet_pascal.caffemodel
fi

if [ -f "segnet_pascal.prototxt" ];
then
  echo "senet50 prototxt file already exists."
else
  echo "Downloading senet50 prototxt file"
  wget -c https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/2d0457ca20a7d22a81f07316bc04b2f26992730c/Example_Models/segnet_pascal.prototxt
fi

cd ${VOC2012_DATASETS_PATH}
if [ ! -d VOCdevkit ];
then
  echo "Downloading VOCtrainval_11-May-2012.tar"
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar    
else
  echo "Datasets already exists."
fi
