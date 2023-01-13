#bin/bash
set -e
set -x

if [ ! -d $DATASETS_PATH ];then
  mkdir -p $DATASETS_PATH
fi 
cd $DATASETS_PATH

if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi

if [ ! -d $MODEL_PATH ];then
  mkdir -p $MODEL_PATH
fi 
cd $MODEL_PATH

if [ ! -d "ResNet50_infer" ];then
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar && tar -xf ResNet50_infer.tar
fi
