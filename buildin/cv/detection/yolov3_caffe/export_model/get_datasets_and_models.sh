#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d $COCO_DATASETS_PATH ];then
  mkdir -p $COCO_DATASETS_PATH
fi 

cd $COCO_DATASETS_PATH

if [ ! -d $FILE1 ];then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
fi

if [ ! -d $FILE2 ];then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
fi

if [ ! -f $FILE3 ];then
  echo "coco.names"
    wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names --no-check-certificate -O coco.names
fi 

if [ ! -d $MODEL_PATH ];then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH

FILE4="yolov3.caffemodel"
FILE5="yolov3.prototxt"
if [ ! -f $FILE4 ];then
  echo $FILE4
  gdown -c https://drive.google.com/uc?id=1mqjMN0KMCB1Yohj0lC-NnnXoBIluw1b2 -O $FILE4
fi 

if [ ! -f $FILE5 ];then
  echo $FILE5
  gdown -c https://drive.google.com/uc?id=1upmVBIxNChy1DE9LJM7dzIQYTodY_P1y -O $FILE5
fi 