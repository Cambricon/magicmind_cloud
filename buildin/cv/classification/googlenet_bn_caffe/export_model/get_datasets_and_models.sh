#bin/bash
set -e
set -x

FILE3=googlenet_bn.caffemodel
FILE4=googlenet_bn_deploy.prototxt

if [ ! -d $MODEL_PATH ];then
    mkdir -p $MODEL_PATH
fi
cd $MODEL_PATH
if [ ! -f "$FILE3" ];then
  echo "$FILE3"
  wget -c  https://github.com/lim0606/caffe-googlenet-bn/blob/master/snapshots/googlenet_bn_stepsize_6400_iter_1200000.caffemodel?raw=true --no-check-certificate -O $FILE3
fi 
if [ ! -f "$FILE4" ];then
  echo "$FILE4"
  wget -c https://raw.githubusercontent.com/lim0606/caffe-googlenet-bn/master/deploy.prototxt --no-check-certificate -O $FILE4
fi

echo "modify $FILE4"
sed -i "s/layers/layer/" $FILE4

if [ ! -d $ILSVRC2012_DATASETS_PATH ];then
    mkdir -p $ILSVRC2012_DATASETS_PATH
fi
cd $ILSVRC2012_DATASETS_PATH
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/ to $ILSVRC2012_DATASETS_PATH"
    exit 1
fi
