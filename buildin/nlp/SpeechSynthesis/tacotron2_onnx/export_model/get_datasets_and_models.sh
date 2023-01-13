#!/bin/bash
set -e
set -x

# download checkpoints
if [ ! -d $PROJ_ROOT_PATH/data ];
then
    mkdir $PROJ_ROOT_PATH/data
fi
if [ ! -d $MODEL_PATH ];
then
    mkdir $MODEL_PATH
fi

cd $MODEL_PATH
if [ -f nvidia_tacotron2pyt_fp16_20190427 ];
then
    echo "tacotron2 checkpoints file already exists."
else
    echo "download checkpoints ..."
    wget -c https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427
fi

if [ -f nvidia_waveglow256pyt_fp16 ];
then
    echo "waveglow checkpoints file already exists."
else
    echo "download checkpoints ..."
    wget -c https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/19.10.0/files/nvidia_waveglow256pyt_fp16
fi
