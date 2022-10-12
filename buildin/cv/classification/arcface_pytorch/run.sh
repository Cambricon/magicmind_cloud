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
bash run.sh qint8_mixed_float16 1 

###3. infer
cd $PROJ_ROOT_PATH/infer_cpp
bash run.sh qint8_mixed_float16 1

###4. compute accuracy top1/top5
cd $PROJ_ROOT_PATH/benchmark
bash eval.sh qint8_mixed_float16 1

###5. benchmark test
cd $PROJ_ROOT_PATH/benchmark
bash perf.sh qint8_mixed_float16 1 1
