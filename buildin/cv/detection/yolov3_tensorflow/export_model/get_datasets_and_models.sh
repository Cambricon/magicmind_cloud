#bin/bash
set -e
set -x

val2017="val2017"
annotations="annotations"
coco_names="coco.names"

if [ ! -d ${COCO_DATASETS_PATH} ];
then
  mkdir -p ${COCO_DATASETS_PATH}
fi 
cd ${COCO_DATASETS_PATH}

if [ ! -d ${val2017} ];
then 
  echo "Downloading val2017.zip"
  wget -c http://images.cocodataset.org/zips/val2017.zip
  unzip -o val2017.zip
fi

if [ ! -d ${annotations} ];
then
  echo "Downloading annotations_trainval2017.zip"
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -o annotations_trainval2017.zip
fi

if [ ! -f ${coco_names} ];
then
  echo "coco.names"
  wget -c https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names --no-check-certificate -O coco.names
fi 

if [ ! -d ${MODEL_PATH} ];
then
  mkdir -p ${MODEL_PATH}
fi 
cd ${MODEL_PATH}

MODEL_TAR_PATH=${MODEL_PATH}/yolov3_coco.tar.gz
#yolov3_coco.ckpt.index
if [ -f ${MODEL_TAR_PATH} ];then 
	index_file=${MODEL_PATH}/yolov3_coco.ckpt.index
	if [ ! -f ${index_file} ];then
        tar -xvf yolov3_coco.tar.gz
	fi
else
    wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
    tar -xvf yolov3_coco.tar.gz
fi
