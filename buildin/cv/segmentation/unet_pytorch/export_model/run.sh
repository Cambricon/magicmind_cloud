#!/bin/bash
set -e
set -x

if [ ! -d ${PROJ_ROOT_PATH}/data ];
then
    mkdir ${PROJ_ROOT_PATH}/data
fi

if [ ! -d ${PROJ_ROOT_PATH}/data/models ];
then
    mkdir ${PROJ_ROOT_PATH}/data/models
fi

if [ -f "${PROJ_ROOT_PATH}/data/models/unet_carvana_scale0.5_epoch2_trace.pt" ];
then echo "unet_carvana_scale0.5_epoch2_trace.pt already exists."
else 
    # 1.下载并安装UNet
    cd ${PROJ_ROOT_PATH}/export_model
    if [ -d "Pytorch-UNet" ];
    then
        echo "UPytorch-UNetet already exists."
    else
        echo "git clone Pytorch-UNet..."
        git clone https://github.com/milesial/Pytorch-UNet.git
    fi
    cd ${PROJ_ROOT_PATH}/export_model/Pytorch-UNet
    git reset --hard  40d5ba797c5689cd9560233baa0e52f28f92727c

    # 2.下载模型和数据集
    cd ${PROJ_ROOT_PATH}/export_model
    bash get_datasets_and_models.sh
    
    # 3.trace model
    cd ${PROJ_ROOT_PATH}/export_model
    python export.py -o ${PROJ_ROOT_PATH}/data/models/unet_carvana_scale0.5_epoch2_trace.pt \
                     -m ${PROJ_ROOT_PATH}/data/models/unet_carvana_scale0.5_epoch2.pth 
                  
fi
