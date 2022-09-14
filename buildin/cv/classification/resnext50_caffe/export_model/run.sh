#!/bin/bash
set -e

# 1.下载数据集
if [ -d $DATASETS_PATH  ];
then 
    echo "ILSVRC2012 datasets already exists."
else
    echo "Please follow the README.md to download the ILSVRC2012 datasets."
    exit -1
fi

# 2.下载caffe模型文件
if [ -f $MODEL_PATH/deploy_resnext50-32x4d.prototxt ] \
   && [ -f $MODEL_PATH/resnext50-32x4d.caffemodel ]	;
then 
    echo "The resnext50 model already exists."
else
    echo "Please follow the README.md to download the Model Zoo in $MODEL_PATH"
    exit -1
fi

