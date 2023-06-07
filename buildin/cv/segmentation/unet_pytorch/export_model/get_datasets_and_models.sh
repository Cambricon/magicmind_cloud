#!/bin/bash
set -e
set -x

cd ${MODEL_PATH}
if [ -f "unet_carvana_scale0.5_epoch2.pth" ];
then
  echo "unet_carvana_scale0.5_epoch2.pth already exists."
else
  echo "Downloading models ..."
  wget -c https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth 
fi

if [ ! -d ${CARVANA_DATASETS_PATH} ];
then
  mkdir ${CARVANA_DATASETS_PATH}
fi

cd ${CARVANA_DATASETS_PATH}
if [ ! -d imgs ] || [ ! -d masks ];
then
  echo "The Carvana data is available on the Kaggle website. "
  echo "You can also download it using the helper script: 
        cd ${PROJ_ROOT_PATH}/export/Pytorch-UNet/
        bash scripts/download_data.sh
        mv /data/imgs and /data/masks to ${CARVANA_DATASETS_PATH}/"
	exit 1
else
  echo "Carvana data already exists."
fi

