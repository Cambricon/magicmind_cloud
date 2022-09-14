#bin/bash
set -e
set -x

ORIGIN_PT='mobilenetv3_small_67.4.pth.tar'
mkdir -p $MODEL_PATH

cd $MODEL_PATH
if [ ! -f $ORIGIN_PT ];then
  echo "mobilenetv3_small_67.4.pth.tar"
  gdown -c https://drive.google.com/uc?id=1lCsN3kWXAu8C30bQrD2JTZ7S2v4yt23C -O $ORIGIN_PT
fi

if [ ! -f "pytorch-mobilenet-v3.zip" ];then
  echo "pytorch-mobilenet-v3.zip"
  wget -c https://github.com/kuan-wang/pytorch-mobilenet-v3/archive/refs/heads/master.zip -O pytorch-mobilenet-v3.zip
  unzip -o pytorch-mobilenet-v3.zip -d $PROJ_ROOT_PATH/export_model
fi

if [ ! -f $DATASETS_PATH/names.txt ];then
  echo "names.txt"
  wget -c https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/inception/imagenet_class_names.txt --no-check-certificate -O $DATASETS_PATH/names.txt
fi 
