#!/bin/bash
set -e
set -x


if [ ! -d $MODEL_PATH ]; then mkdir -p $MODEL_PATH; fi

cd $MODEL_PATH
if [ ! -f "Conformer_small_patch16.pth" ];
then
  echo "Please download Conformer_small_patch16.pth from https://pan.baidu.com/share/init?surl=kYOZ9mRP5fvujH6snsOjew [code:qvu8]"
  exit 1
fi

cd $DATASETS_PATH
if [ ! -d val ];
then
    echo "Download and extract ImageNet val images from http://image-net.org/."
    echo "the validation data is expected to be in the val folder:"
    echo "/path/to/imagenet/"
    echo "  val/"
    echo "    class1/"
    echo "      img1.jpeg"
    echo "reference for data preparation：https://github.com/onnx/models/blob/main/vision/classification/imagenet_prep.md"
    exit 1
fi


