#bin/bash
set -e
set -x

FILE1="val2017"
FILE2="annotations"
FILE3="coco.names"
FILE4="mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"

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

if [ ! -f $MODEL_PATH/$FILE4 ];then
  echo "mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
  wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth --no-check-certificate -O mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
fi 

cd $PROJ_ROOT_PATH/export_model
if [ ! -d "mmdetection" ];then
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    git checkout 73b4e65a6a30435ef6a35f405e3474a4d9cfb234
    git apply ../magicmind.patch
else
    echo "mmdetection exist!"
fi

cd $PROJ_ROOT_PATH/export_model/mmdetection
pip install mmcv-full==1.6.0 tabulate
pip install -e .
