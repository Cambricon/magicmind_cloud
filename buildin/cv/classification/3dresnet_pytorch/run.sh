#!/bin/bash
set -e
set -x
precision=force_float32
batch_size=1

# 0. export model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

# 1. gen model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $batch_size
# 2. infer & eval
cd $PROJ_ROOT_PATH/infer_python
bash run.sh $precision
