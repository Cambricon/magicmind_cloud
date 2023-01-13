#!/bin/bash
set -e
set -x

# 0. convert model
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision batch_size
bash run.sh qint8_mixed_float16 1

# 2 build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh precision batch_size
bash run.sh qint8_mixed_float16 1
