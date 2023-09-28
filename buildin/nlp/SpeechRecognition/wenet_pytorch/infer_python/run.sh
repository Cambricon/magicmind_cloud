#!/bin/bash
set -e
set -x

# Copyright 2022 Cambricon, Inc. All Rights Reserved.

encoder_magicmind_model=$1
decoder_magicmind_model=$2
s0_path=${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0

infer_res_dir="$PROJ_ROOT_PATH/data/output/$(basename ${decoder_magicmind_model})_infer_res"
mkdir -p $PROJ_ROOT_PATH/data/output/
if [ ! -f ${infer_res_dir} ];
then
    touch $infer_res_dir
fi

# attention_rescoring
echo $decoder_magicmind_model
python infer.py  --device_id 0 \
                 --config     ${s0_path}/conf/train_conformer.yaml \
                 --test_data  ${s0_path}/data/local/test/data.list \
                 --dict       ${MODEL_PATH}/20211025_conformer_exp/words.txt \
                 --mode       attention_rescoring \
                 --encoder_magicmind $encoder_magicmind_model  \
                 --decoder_magicmind $decoder_magicmind_model  \
                 --result_file $infer_res_dir


python ${PROJ_ROOT_PATH}/export_model/wenet/tools/compute-wer.py --char=1 --v=0 ${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0/data/test/text  $infer_res_dir


