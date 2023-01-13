#!/bin/bash
set -e
set -x
BATCH_SIZE=$1
MAX_SEQ_LENGTH=$2
if [ -f "$PROJ_ROOT_PATH/data/models/roberta_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.onnx" ];
then
  echo "$PROJ_ROOT_PATH/data/models/roberta_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.onnx already exists."
else
  echo "generate $PROJ_ROOT_PATH/data/models/roberta_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.onnx"

  # 1.下载数据集和模型
  cd $PROJ_ROOT_PATH/export_model
  bash get_datasets_and_models.sh
  

  
  
  # 2.安装transformers
  cd $PROJ_ROOT_PATH/export_model
  pip install transformers
  
  # 3. export roberta.onnx
  python export.py --model_path $MODEL_PATH/chinese-roberta-wwm-ext-chnsenticorp \
                   --onnx_model $PROJ_ROOT_PATH/data/models/roberta_${BATCH_SIZE}bs_${MAX_SEQ_LENGTH}.onnx \
		   --batch_size ${BATCH_SIZE} \
                   --max_seq_length ${MAX_SEQ_LENGTH}
fi
