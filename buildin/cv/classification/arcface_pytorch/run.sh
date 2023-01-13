#!/bin/bash
set -e
set -x
echo "Start !"

source env.sh

###1. convert torch.jit.trace model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###2. build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision batch_size
bash run.sh qint8_mixed_float16 1 

###3. infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh precision batch_size image_num
bash run.sh qint8_mixed_float16 1 1000
