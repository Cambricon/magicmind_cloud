#!/bin/bash
set -e
set -x

# 0. convert model
pip install -r requirement.txt
cd $PROJ_ROOT_PATH/export_model 
bash run.sh

# 1. gen_model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh quant_mode shape_mutable batch_size conf iou max_det
bash run.sh force_float32 true 1 0.001 0.65 1000

# 2. build infer_cpp and infer
cd $PROJ_ROOT_PATH/infer_cpp
#bash run.sh quant_mode shape_mutable batch_size batch image_num(-1: all images)
bash run.sh force_float32 true 1 1 -1

### 3.eval and perf
#bash $PROJ_ROOT_PATH/benchmark/eval.sh quant_mode shape_mutable batch_size image_num(-1: all images)
bash $PROJ_ROOT_PATH/benchmark/eval.sh force_float32 true 1 -1
#bash $PROJ_ROOT_PATH/benchmark/perf.sh quant_mode shape_mutable batch_size batch threads
bash $PROJ_ROOT_PATH/benchmark/perf.sh force_float32 true 1 1 1
