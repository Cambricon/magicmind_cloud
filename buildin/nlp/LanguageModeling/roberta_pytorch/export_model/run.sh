#!/bin/bash
set -e
set -x
onnx_path=${MODEL_PATH}/roberta_1bs_128.onnx
BATCH_SIZE=${1:-1}
MAX_SEQ_LENGTH=${2:-128}
ONNX_MODEL=${3:-${onnx_path}}

if [ $# -gt 3 ];
then
    echo "bash run.sh <batch_size> <max_seq_length> <onnx_path>"
    echo "Usage: bash run.sh 1 128 ../data/models/roberta_1bs_128.onnx"
    exit -1
fi

if [ -f ${ONNX_MODEL} ];
then
  echo "${ONNX_MODEL} already exists."
else
  echo "generate ${ONNX_MODEL}"

  # 1.下载数据集和模型
  cd ${PROJ_ROOT_PATH}/export_model
  bash get_datasets_and_models.sh
  if [ ! $? -eq 0 ]; then
    exit -1
  fi
  
  # 2.安装transformers
  cd ${PROJ_ROOT_PATH}/export_model
  pip install transformers
  
  # 3. export roberta.onnx
  python export.py --model_path ${MODEL_PATH}/chinese-roberta-wwm-ext-chnsenticorp \
                   --onnx_model ${ONNX_MODEL} \
		               --batch_size ${BATCH_SIZE} \
                   --max_seq_length ${MAX_SEQ_LENGTH}
fi
