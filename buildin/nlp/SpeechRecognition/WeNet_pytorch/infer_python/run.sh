#!/bin/bash

# Copyright 2022 Cambricon, Inc. All Rights Reserved.

PRECISION=$1 #force_float32
s0_path=${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0

if [ ! -d $PROJ_ROOT_PATH/data/output ];
then
    mkdir -p $PROJ_ROOT_PATH/data/output
fi

if [ ! -f ${MODEL_PATH}/encoder_${PRECISION}_model ];then
    echo "$encoder_${PRECISION}_model not exist,please go to gen_model folder and run this command:bash run.sh ${PRECISION}!!!"
    exit 1
fi

if [ ! -f ${MODEL_PATH}/decoder_${PRECISION}_model ];then
    echo "$encoder_${PRECISION}_model not exist,please go to gen_model folder and run this command:bash run.sh ${PRECISION}!!!"
    exit 1
fi


# attention_rescoring
python infer.py \
  --config     ${s0_path}/conf/train_conformer.yaml \
  --test_data  ${s0_path}/data/local/test/data.list \
  --dict       ${MODEL_PATH}/20211025_conformer_exp/words.txt \
  --mode       attention_rescoring \
  --encoder_magicmind ${MODEL_PATH}/encoder_${PRECISION}_model  \
  --decoder_magicmind ${MODEL_PATH}/decoder_${PRECISION}_model  \
  --result_file ${PROJ_ROOT_PATH}/data/output/infer_python_output_${PRECISION}


python ${PROJ_ROOT_PATH}/export_model/wenet/tools/compute-wer.py --char=1 --v=0 ${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0/data/test/text  ${PROJ_ROOT_PATH}/data/output/infer_python_output_${PRECISION}  2>&1 | tee $PROJ_ROOT_PATH/data/output/${PRECISION}_log_eval


