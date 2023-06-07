#!/bin/bash
set -e
set -x

MAX_SEQ_LENGTH=$1

# 下载bert-base-cased初始权重
if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir $PROJ_ROOT_PATH/data/models
fi
cd $MODEL_PATH
if [ -f "bert_squad_pytorch_${MAX_SEQ_LENGTH}.pt" ]; 
then
  echo "traced pt already exists."
else
  if [ -d "bert-squad-training" ]; 
  then
    echo "model file already exists."
  else
    echo "Downloading model file"
    git clone https://huggingface.co/linfuyou/bert-squad-training.git
    cd $MODEL_PATH/bert-squad-training
    rm pytorch_model.bin
    wget https://huggingface.co/linfuyou/bert-squad-training/resolve/main/pytorch_model.bin
  fi
fi

# 下载测试数据集
cd $SQUAD_DATASETS_PATH
if [ -f "dev-v1.1.json" ];
then 
  echo "dev-v1.1.json already exists."
else
  echo "Downloading dev-v1.1.json"
  wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi
