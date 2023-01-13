#!/bin/bash
set -e
set -x

cd $DATASETS_PATH
# 1.下载数据集
if [ ! -d "WIDER_val" ];
then 
  echo "Downloading WIDER_val.zip"
  gdown -c https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q -O WIDER_val.zip
  unzip -o WIDER_val.zip
else 
  echo "WIDER_val already exists."
fi

# 2.下载权重文件
mkdir -p $PROJ_ROOT_PATH/data/models
cd $PROJ_ROOT_PATH/data/models
if [ -f "Resnet50_Final.pth" ];
then 
    echo "Resnet50_Final.pth already exists."
else 
    echo "Downloading Resnet50_Final.pth file"
    gdown -c https://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW -O Resnet50_Final.pth
fi
