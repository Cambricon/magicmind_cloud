#bin/bash
set -e
set -x

if [ ! -d ${ILSVRC2012_DATASETS_PATH} ];then
  mkdir -p ${ILSVRC2012_DATASETS_PATH}
fi 
cd ${ILSVRC2012_DATASETS_PATH}

if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi

if [ ! -d ${MODEL_PATH} ];then
  mkdir -p ${MODEL_PATH}
fi 
cd ${MODEL_PATH}

if [ ! -d "EfficientNetB7_infer" ];then
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/EfficientNetB7_infer.tar && tar -xf EfficientNetB7_infer.tar
fi
