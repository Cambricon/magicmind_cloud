#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"
FILE4="yolov3_tiny.cfg"
FILE5="yolov3_tiny.weights"
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

FILE6="yolov3_tiny.caffemodel"
FILE7="yolov3_tiny.prototxt"
if [ ! -f $FILE6 ];then
  echo $FILE6
  gdown -c https://drive.google.com/uc?id=1zz6Dq4hdUNMVmE_aLZ5bQzc80vfNv4aN -O $FILE6
fi 
if [ ! -f $FILE7 ];then
  echo $FILE7
  gdown -c https://drive.google.com/uc?id=1I_vSuiDslsCzJf_TvP1IqdN7M5pBxMMH -O $FILE7
fi