#!/bin/bash

cd $MODEL_PATH
if [ -f "frozen_graph.pb" ]; 
then
  echo "model file already exists."
else
  echo "Please go to https://github.com/NVIDIA/FasterTransformer/tree/v3.0, get the model through step by step"
  exit -1
fi

# 下载测试数据集
cd $DATASETS_PATH
if [ -f "dev-v1.1.json" ];
then 
  echo "dev-v1.1.json already exists."
else
  echo "Downloading dev-v1.1.json"
  wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi
