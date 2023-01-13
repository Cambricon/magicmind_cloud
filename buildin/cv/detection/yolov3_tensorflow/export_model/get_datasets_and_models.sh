#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d $DATASETS_PATH ];
then
  mkdir -p $DATASETS_PATH
fi 
cd $DATASETS_PATH

if [ ! -d $FILE1 ];
then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
fi

if [ ! -d $FILE2 ];
then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
fi

if [ ! -f $FILE3 ];
then
  echo "coco.names"
  wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names --no-check-certificate -O coco.names
fi 

if [ ! -d $MODEL_PATH ];
then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH

MODEL_TAR_PATH=$MODEL_PATH/yolov3_coco.tar.gz
if [ -f $MODEL_TAR_PATH ];
then 
    tar -xvf yolov3_coco.tar.gz
else
    wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
    tar -xvf yolov3_coco.tar.gz
fi