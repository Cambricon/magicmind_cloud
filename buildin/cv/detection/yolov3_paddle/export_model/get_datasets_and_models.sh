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

if [ -d "PaddleDetection" ];then
  rm -rf PaddleDetection
fi
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
git checkout v2.5.0
if grep -q "#  nms:" /$MODEL_PATH/PaddleDetection/configs/yolov3/_base_/yolov3_darknet53.yml;
then
  echo "modifying the paddle yolov3 has been already done"
else
  echo "modifying the paddle yolov3..."
  git apply $PROJ_ROOT_PATH/export_model/yolov3_paddle.patch
fi
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o use_gpu=false weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams TestReader.inputs_def.image_shape=[3,608,608] --output_dir inference_model_nonms
