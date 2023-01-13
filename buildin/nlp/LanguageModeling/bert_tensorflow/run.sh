#!/bin/bash
set -e
set -x

###0. convert model
cd $PROJ_ROOT_PATH/export_model
#bash run.sh batch_size max_seq_len
bash run.sh

###1. gen_model - qint8_mixed_float16  force_float16 force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size max_seq_len
bash run.sh force_float32 true 1 384 

###2. infer_python
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision shape_mutable batch_size max_seq_len
bash run.sh force_float32 true 1 384
