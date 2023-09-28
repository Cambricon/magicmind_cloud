#!/bin/bash
set -e

# 1.下载数据集
if [ -d $ILSVRC2012_DATASETS_PATH  ];
then 
    echo "ILSVRC2012 datasets already exists."
else
    echo "Please use the URL in the end of README.md to download ILSVRC2012"
    exit -1
fi

# 2.下载caffe模型文件
if [ -f $MODEL_PATH/deploy_resnext50-32x4d.prototxt ] \
   && [ -f $MODEL_PATH/resnext50-32x4d.caffemodel ]	;
then 
    echo "The resnext50 model already exists."
else
    # The model can be downloaded from the first link in the README.md
    echo "Please use the URL in the end of README.md to download the Model, and move it to $MODEL_PATH"
    exit -1
fi

