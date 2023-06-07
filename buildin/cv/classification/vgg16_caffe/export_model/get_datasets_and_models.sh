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
if [ -f "vgg16.caffemodel" ];
then
  echo "vgg16 caffemodel already exists."
else
  echo "Downloading vgg16 caffemodel file"
  wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O vgg16.caffemodel
fi

if [ -f "deploy.prototxt" ];
then
  echo "vgg16 prototxt file already exists."
else
  echo "Downloading vgg16 prototxt file"
  wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt \
    -O deploy.prototxt
fi

cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi


