#!/bin/bash
set -e
set -x

# 下载bert-base-cased初始权重
if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir $PROJ_ROOT_PATH/data/models
fi
cd $MODEL_PATH
if [ -d "pytorch_bert_base_cased_squad" ]; 
then
  echo "model file already exists."
else
  echo "Downloading model file"
  wget http://gitlab.software.cambricon.com/neuware/software/solutionsdk/pytorch_bert_base_cased_squad_pretrained/-/blob/master/pytorch_bert_base_cased_squad.tgz
  tar -zvxf pytorch_bert_base_cased_squad.tgz
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
