#bin/bash
set -e
set -x

FILE1="mobilenet_v2.caffemodel"
FILE2="mobilenet_v2_deploy.prototxt"

if [ ! -d ${MODEL_PATH} ];then
    mkdir -p ${MODEL_PATH}
fi
cd ${MODEL_PATH}
if [ ! -f ${FILE1} ];then
  echo "mobilenet_v2.caffemodel"
  wget -c https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel --no-check-certificate -O mobilenet_v2.caffemodel
fi 
if [ ! -f ${FILE2} ];then
  echo "mobilenet_v2_deploy.prototxt"
  wget -c https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt --no-check-certificate -O mobilenet_v2_deploy.prototxt
fi 

if [ ! -d ${ILSVRC2012_DATASETS_PATH} ];then
    mkdir -p ${ILSVRC2012_DATASETS_PATH}
fi
cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/ to $ILSVRC2012_DATASETS_PATH"
    exit 1
fi
