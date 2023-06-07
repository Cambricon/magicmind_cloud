#!/bin/bash
set -e

EXPORT(){
  echo "export model begin..."
  python ${PROJ_ROOT_PATH}/export_model/export.py --weights ${MODEL_PATH}/backbone.pth --output_pt ${MODEL_PATH}/arcface_r100.pt
  echo "export model end..."
}

# 1.下载数据集
if [ -d ${IJB_DATASETS_PATH}  ];
then 
    echo "IJB datasets already exists."
else
    echo "Please follow the README.md to download the IJB-B,IJB-C datasets."
    exit -1
fi

# 2.下载权重文件
if [ -f ${MODEL_PATH}/backbone.pth ];
then 
    echo "The arcface backborn already exists."
else
    echo "Please follow the README.md to download the Model Zoo in ${MODEL_PATH}"
    exit -1
fi

# 3.trace model
# param: batchsize
cd ${PROJ_ROOT_PATH}/export_model
if [ -f ${MODEL_PATH}/arcface_r100.pt ];then
    echo "arcface_r100.pt aleady exists."
else
    if [ ! -d "insightface" ];
    then
      wget -c https://github.com/deepinsight/insightface/archive/478aafb4fc66030e07fb46143e8e069f85e68147.zip -O insightface.zip
      unzip insightface.zip
      mv insightface-478aafb4fc66030e07fb46143e8e069f85e68147 insightface
    else
      echo "insightface ready"
    fi

    EXPORT 
fi

