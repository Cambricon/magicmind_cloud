#!/bin/bash
set -e
set -x


if [ ! -d $MODEL_PATH ]; then mkdir -p $MODEL_PATH; fi

cd $MODEL_PATH
if [ ! -f d2_224_85.2.pth.tar ];
then
  wget https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar
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
