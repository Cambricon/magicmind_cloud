#!/bin/bash
set -e
set -x
precision=force_float16
batch_size_min=1
batch_size=4
batch_size_max=16
input_len=128

###0. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###1. gen_model 
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $batch_size_min $batch_size $batch_size_max

###2. infer_python and eval
cd $PROJ_ROOT_PATH/infer_python
bash run.sh $precision $batch_size_min $batch_size $batch_size_max $input_len
