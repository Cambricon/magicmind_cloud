#!/bin/bash
set -e
set -x

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exists"
else
    mkdir $PROJ_ROOT_PATH/data/models
fi

cd $MODEL_PATH
if [ -f "squeezenet_v1_1.caffemodel" ];
then
  echo "squeezenet_v1.1 caffemodel already exists."
else
  echo "Downloading squeezenet_v1.1 caffemodel file"
  wget -c https://raw.githubusercontent.com/forresti/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel -O squeezenet_v1_1.caffemodel
fi

cd $MODEL_PATH
if [ -f "deploy_v1_1.prototxt" ];
then
  echo "squeezenet prototxt file already exists."
else
  echo "Downloading squeezenet prototxt file"
  wget -c https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/deploy.prototxt -O deploy_v1_1.prototxt
fi

cd $DATASETS_PATH
if [ ! -d images ];
then
    echo "Downloading LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/ to $DATASETS_PATH"
    exit 1
fi
