#!/bin/bash
set -e
set -x

if [ -d $PROJ_ROOT_PATH/data/models ];
then
    echo "folder $PROJ_ROOT_PATH/data/models already exists."
else
    mkdir $PROJ_ROOT_PATH/data/models
fi

cd $MODEL_PATH
if [ -f "mobilenet_iter_73000.caffemodel" ];
then
  echo "ssd caffemodel already exists."
else
  echo "Downloading ssd caffemodel file"
  wget -c https://github.com/chuanqi305/MobileNet-SSD/raw/97406996b1eee2d40eb0a00ae567cf41e23369f9/mobilenet_iter_73000.caffemodel
fi

cd $MODEL_PATH
if [ -f "deploy.prototxt" ];
then
  echo "ssd prototxt file already exists."
else
  echo "Downloading ssd prototxt file"
  wget -c https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt
fi

cd $DATASETS_PATH
if [ -f "VOCtrainval_11-May-2012.tar" ];
then
  echo "voc2012 datasets already exists."
else
  echo "Downloading voc2012 datasets..."
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_11-May-2012.tar
fi

cd $UTILS_PATH
if [ -f "compute_voc_mAP.py" ];
then
    echo "compute_voc_mAP.py already exists"
else
    echo "Downloading compute_voc_mAP.py"
    wget https://raw.githubusercontent.com/luliyucoordinate/eval_voc/361b1953891827b2342b6d6ce92b66a31855cb0e/eval_voc.py -O compute_voc_mAP.py
    patch -u $UTILS_PATH/compute_voc_mAP.py -i $PROJ_ROOT_PATH/export_model/eval_voc.diff
fi
