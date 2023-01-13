#!/bin/bash
set -e
set -x

BATCH_SIZE=$1
MAX_SEQ_LENGTH=$2
if [ -f "$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt" ];
then
  echo "$PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt already exists."
else
  echo "generate $PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt"

  # 1.下载数据集和模型
  cd $PROJ_ROOT_PATH/export_model
  bash get_datasets_and_models.sh
  
  # 2.下载transformer v3.1.0实现源码
  cd $PROJ_ROOT_PATH/export_model
  
  # 3.安装transformers
  pip install transformers==3.1.0
  
  # 4. export bert_squad.pt
  python export.py --model_path $MODEL_PATH/bert-squad-training \
                   --pt_model $PROJ_ROOT_PATH/data/models/bert_squad_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt \
                   --batch_size ${BATCH_SIZE} \
                   --max_seq_length ${MAX_SEQ_LENGTH}
fi
