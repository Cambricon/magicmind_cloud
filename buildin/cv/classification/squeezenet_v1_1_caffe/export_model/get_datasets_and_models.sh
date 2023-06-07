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
if [ -f "squeezenet_v1_1.caffemodel" ];
then
  echo "squeezenet_v1.1 caffemodel already exists."
else
  echo "Downloading squeezenet_v1.1 caffemodel file"
  wget -c https://raw.githubusercontent.com/forresti/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel -O squeezenet_v1_1.caffemodel
fi

if [ -f "deploy_v1_1.prototxt" ];
then
  echo "squeezenet prototxt file already exists."
else
  echo "Downloading squeezenet prototxt file"
  wget -c https://raw.githubusercontent.com/forresti/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt -O deploy_v1_1.prototxt
fi

cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi

