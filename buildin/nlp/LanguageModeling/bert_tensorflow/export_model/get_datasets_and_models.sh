#!/bin/bash
set -e
set -x
if [ -d $MODEL_PATH ];
then
    echo "folder $MODEL_PATH already exist!!!"
else
    mkdir -p "$MODEL_PATH"
fi

if [ -d $SQUAD_DATASETS_PATH ];
then
    echo "folder $SQUAD_DATASETS_PATH already exist!!!"
else
    mkdir -p "$SQUAD_DATASETS_PATH"
fi

# 下载测试数据集
file_path=$SQUAD_DATASETS_PATH/dev-v1.1.json
if [ -f $file_path ];
then 
  echo "dev-v1.1.json already exists."
else
  echo "Downloading dev-v1.1.json"
  wget -P $SQUAD_DATASETS_PATH https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
fi

bert_tf_path=$MODEL_PATH/bert_tf_v1_1_base_fp32_384_2.zip
#下载模型
if [ -f ${bert_tf_path} ]; 
then
  echo "bert tf file already exists."
else
  # 模型下载地址
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_v1_1_base_fp32_384/versions/2/zip \
       -O ${bert_tf_path}
  unzip -o ${bert_tf_path} -d ${MODEL_PATH}/squad_model
fi