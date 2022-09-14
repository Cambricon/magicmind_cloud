#!/bin/bash
set -e
set -x

BATCH_SIZE=$1
MAX_SEQ_LENGTH=$2
if [ -f "$PROJ_ROOT_PATH/data/models/bert_qa_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt" ];
then
  echo "$PROJ_ROOT_PATH/data/models/bert_qa_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt already exists."
else
  echo "generate $PROJ_ROOT_PATH/data/models/bert_qa_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt"

  # 1.下载数据集和模型
  cd $PROJ_ROOT_PATH/export_model
  bash get_datasets_and_models.sh
  
  # 2.下载transformer v3.1.0实现源码
  cd $PROJ_ROOT_PATH/export_model
  if [ -d "transformers-3.1.0" ];
  then
    echo "transformers-3.1.0 already exists."
  else
    echo "git clone transformers-3.1.0..."
    wget -c https://github.com/huggingface/transformers/archive/refs/tags/v3.1.0.zip
    unzip -o v3.1.0.zip
  fi
  
  # 3.patch-transformers
  # 由于MagicMind PyTorch模型转换功能暂不支持BERT网络用到的``torch.nn.LayerNorm``和``torch.nn.functional.gelu``。
  if grep -q "Implementation of the gelu activation function" $PROJ_ROOT_PATH/export_model/transformers-3.1.0/src/transformers/modeling_bert.py;
  then
    echo "gelu activation function already exists."
  else 
    echo "add gelu activation function in $PROJ_ROOT_PATH/export_model/transformers-3.1.0/src/transformers/modeling_bert.py"
    cd $PROJ_ROOT_PATH/export_model
    patch transformers-3.1.0/src/transformers/modeling_bert.py < modeling_bert_v3.1.0.patch
  fi
  
  # 4.安装transformers
  cd $PROJ_ROOT_PATH/export_model
  pip install ./transformers-3.1.0
  
  # 5. export bert_qa.pt
  python export.py --model_path $MODEL_PATH/pytorch_bert_base_cased_squad \
                   --pt_model $PROJ_ROOT_PATH/data/models/bert_qa_pytorch_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.pt \
                   --batch_size ${BATCH_SIZE} \
                   --max_seq_length ${MAX_SEQ_LENGTH}
fi
