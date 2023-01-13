#!/bin/bash
set -e
set -x

source env.sh
#1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh 1 128
echo "DOWNLOAD_DATA_SUCCESS!"

#2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 true 1 128
echo "GENERATE MODEL SUCCESS!"

#3.infer_python
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32 true 32 128
echo "INFER PYTHON SUCCESS!"
