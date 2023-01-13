#!/bin/bash
set -e
set -x

###0. convert model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

###1. gen_model - force_float32
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision
bash run.sh force_float32

###2. infer_python and eval
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision
bash run.sh force_float32

python ${PROJ_ROOT_PATH}/export_model/wenet/tools/compute-wer.py --char=1 --v=0 ${PROJ_ROOT_PATH}/export_model/wenet/examples/aishell/s0/data/test/text  ${PROJ_ROOT_PATH}/data/output/infer_python_output_force_float32  2>&1 | tee $PROJ_ROOT_PATH/data/output/force_float32_log_eval

