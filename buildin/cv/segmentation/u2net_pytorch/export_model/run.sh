#!/bin/bash
set -e
set -x

if [ ! -d $MODEL_PATH ];
then
    mkdir -p "$MODEL_PATH"
fi

if [ ! -d $MSRA_B_DATASETS_PATH ];
then
    echo "Please download MSRA-B datasets from https://mmcheng.net/msra10k/"
    exit -1
fi

if [ ! -f $MODEL_PATH/u2net.pth ];
then
    echo "Please download u2net.pth from https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing"
    exit -1
fi

if [ ! -f $MODEL_PATH/u2net.pt ];
then
    cd $PROJ_ROOT_PATH/export_model
    if [ ! -d U-2-Net ];
    then
        git clone https://github.com/xuebinqin/U-2-Net.git
        cd U-2-Net
        git checkout 53dc9da026650663fc8d8043f3681de76e91cfde
        cd ..
    fi
    python export.py
fi
