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
if [ -d "deeplabv3_mnv2_pascal_train_aug" ];
then
  echo "deeplabv3 model already exists."
else
  echo "Downloading deeplabv3 model file"
  wget -c http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz 
  tar -zxvf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
fi

cd $VOC2012_DATASETS_PATH
if [ ! -d VOCdevkit ];
then
  echo "Downloading VOCtrainval_11-May-2012.tar"
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar    
else
  echo "Datasets already exists."
fi
