#!/bin/bash
set -e
set -x

if [ -d $MODEL_PATH ];
then
    echo "folder $MODEL_PATH already exists"
else
    mkdir -p $MODEL_PATH
fi

cd $MODEL_PATH
if [ -f "./hub/checkpoints/swin_t-704ceda3.pth" ];
then
  echo "swin_t.pth already exists."
else
  echo "Downloading pretrained swin_t.pth file"
  wget https://download.pytorch.org/models/swin_t-704ceda3.pth  --directory-prefix=./hub/checkpoints/
fi

cd ${ILSVRC2012_DATASETS_PATH}
if [ ! -f ILSVRC2012_val_00000001.JPEG ];
then
    echo "Please download LSVRC_2012_img_val datasets on https://image-net.org/challenges/LSVRC/"
    exit 1
fi
