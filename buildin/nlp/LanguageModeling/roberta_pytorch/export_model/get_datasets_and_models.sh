#/bin/bash
set -e
set -x

# 下载chinese-roberta-wwm-ext初始权重
if [ ! -d $PROJ_ROOT_PATH/data/models ];
then
    mkdir $PROJ_ROOT_PATH/data/models
fi
cd $MODEL_PATH
if [ -d "chinese-roberta-wwm-ext-chnsenticorp" ]; 
then
  echo "model file already exists."
else
  echo "Downloading model file"
  git clone https://huggingface.co/linfuyou/chinese-roberta-wwm-ext-chnsenticorp.git
  cd $MODEL_PATH/chinese-roberta-wwm-ext-chnsenticorp
  rm pytorch_model.bin
  wget https://huggingface.co/linfuyou/chinese-roberta-wwm-ext-chnsenticorp/resolve/main/pytorch_model.bin
fi

# 下载测试数据集
cd $DATASETS_PATH
if [ -f "test.tsv" ];
then 
  echo "test.tsv already exists."
else
  echo "Downloading test.tsv"
  wget https://raw.githubusercontent.com/pengming617/bert_classification/master/data/test.tsv
fi
