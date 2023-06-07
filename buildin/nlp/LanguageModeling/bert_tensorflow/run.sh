#!/bin/bash
set -e
set -x

###0. convert model
cd $PROJ_ROOT_PATH/export_model
#bash run.sh
bash run.sh

###1. gen_model - force_float16 force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <magicmind_model> <precision> <batch_size> <shape_mutable> <max_seq_length> 
bash run.sh ${MODEL_PATH}/bert_fp32_1_false_384 force_float32 1 false 384 

###2. infer_python
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <magicmind_model> <batch_size> <max_seq_length>
bash run.sh ${MODEL_PATH}/bert_fp32_1_false_384 1 384
