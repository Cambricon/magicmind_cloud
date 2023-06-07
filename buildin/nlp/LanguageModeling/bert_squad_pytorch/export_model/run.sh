#!/bin/bash
set -e
set -x

MAX_SEQ_LENGTH=$1

if [ ! -d ${MODEL_PATH} ];then
  mkdir -p ${MODEL_PATH}
fi

TORCH_PT="${MODEL_PATH}/bert_squad_pytorch_${MAX_SEQ_LENGTH}.pt"

if [ -f ${TORCH_PT} ];
then
  echo "${TORCH_PT} already exists."
else
  echo "generate ${TORCH_PT}"

  # 1.下载数据集和模型
  cd $PROJ_ROOT_PATH/export_model
  bash get_datasets_and_models.sh ${MAX_SEQ_LENGTH}
  
  # 2.下载transformer v3.1.0实现源码
  cd $PROJ_ROOT_PATH/export_model
  
  # 3.安装transformers
  pip install transformers==3.1.0
  
  if [ ! -f ${TORCH_PT} ]; 
  then
    python export.py --model_path ${MODEL_PATH}/bert-squad-training \
                   --pt_model ${TORCH_PT} \
                   --max_seq_length ${MAX_SEQ_LENGTH}
  fi
fi
