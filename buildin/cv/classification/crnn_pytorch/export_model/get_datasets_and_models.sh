#!/bin/bash
set -e
set -x

cd ${MODEL_PATH}
if [ -f crnn.pth ];
then
  echo "crnn pth already exists."
else
  echo "Please download crnn pth from https://pan.baidu.com/s/1pLbeCND to ${MODEL_PATH}."
  exit 1
fi

cd ${SYNTH_DATASETS_PATH}
if [ ! -d mnt ];
then
  echo "Please download mjsynth from https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth"
  exit 1
fi

if [ ! -f annotation.txt ];
then
  python data_prepare.py --image_dir ${SYNTH_DATASETS_PATH}/mnt/ramdisk/max/90kDICT32px/
  mv annotation.txt ${SYNTH_DATASETS_PATH}
fi
