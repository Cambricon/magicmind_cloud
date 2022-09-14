#bin/bash
set -e
set -x

FILE3="mobilenet_v2.caffemodel"
FILE4="mobilenet_v2_deploy.prototxt"
mkdir -p $MODEL_PATH
cd $MODEL_PATH
if [ ! -f $FILE3 ];then
  echo "mobilenet_v2.caffemodel"
  wget -c https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel --no-check-certificate -O mobilenet_v2.caffemodel
fi 
if [ ! -f $FILE4 ];then
  echo "mobilenet_v2_deploy.prototxt"
  wget -c https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt --no-check-certificate -O mobilenet_v2_deploy.prototxt
fi 
FILE1='names.txt'
if [ ! -f $DATASETS_PATH/$FILE1 ];then
  echo "names.txt"
  wget -c https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/inception/imagenet_class_names.txt --no-check-certificate -O $DATASETS_PATH/names.txt
fi 
