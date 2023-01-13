#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"

if [ ! -d $DATASETS_PATH ];then
  mkdir -p $DATASETS_PATH
fi 
cd $DATASETS_PATH

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

FILE4="yolov4.caffemodel"
FILE5="yolov4.prototxt"
if [ ! -f $FILE4 ];then
  echo $FILE4
  gdown -c https://drive.google.com/uc?id=1npdUK32pHqgtkeIfO0EuhBU-SEyjTkDi -O $FILE4
fi 

if [ ! -f $FILE5 ];then
  echo $FILE5
  gdown -c https://drive.google.com/uc?id=1_ARfjIOMZiZ_kS6coqyW3wznGWpBLTMZ -O $FILE5
fi 
