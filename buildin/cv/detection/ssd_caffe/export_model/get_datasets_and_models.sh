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
  rm -rf deploy.prototxt
fi

echo "Downloading ssd prototxt file"
wget -c https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt
patch -p0 deploy.prototxt < $PROJ_ROOT_PATH/export_model/prototxt.diff

cd $VOC2007_DATASETS_PATH
if [ -f "VOCtest_06-Nov-2007.tar" ];
then
  echo "voc2007 test datasets already exists."
else
  echo "Downloading voc2007 test datasets..."
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
fi

cd $UTILS_PATH
if [ -f "$UTILS_PATH/compute_voc_mAP.py" ];
then
  rm $UTILS_PATH/compute_voc_mAP.py
fi
echo "Downloading compute_voc_mAP.py"
wget https://raw.githubusercontent.com/luliyucoordinate/eval_voc/361b1953891827b2342b6d6ce92b66a31855cb0e/eval_voc.py -O compute_voc_mAP.py
patch -p0 $UTILS_PATH/compute_voc_mAP.py < $PROJ_ROOT_PATH/export_model/eval_voc.diff
