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
if [ -f "DenseNet_121.caffemodel" ]	;
then 
    echo "The densenet121 model already exists."
else
    echo "Downloading DenseNet_121.caffemodel file."
    gdown -c https://drive.google.com/uc?id=0B7ubpZO7HnlCcHlfNmJkU2VPelE
fi

if [ -f "DenseNet_121.prototxt" ]
then 
    echo "The densenet121 prototxt already exists."
else
    echo "Downloading DenseNet_121.prototxt file."
    wget -c https://raw.githubusercontent.com/shicai/DenseNet-Caffe/master/DenseNet_121.prototxt
fi


cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
else 
    echo "ILSVRC2012 datasets already exists."
fi
