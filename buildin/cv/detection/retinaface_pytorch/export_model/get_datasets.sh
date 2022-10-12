#!/bin/bash
set -e
set -x

cd $DATASETS_PATH

if [ ! -d "WIDER_val" ];
then 
  echo "Downloading WIDER_val.zip"
  gdown -c https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q -O WIDER_val.zip
  unzip -o WIDER_val.zip
else 
  echo "WIDER_val already exists."
fi

