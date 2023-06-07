#!/bin/bash
set -e
set -x
echo "Start!"

source env.sh
###1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

###2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh <precision> <shape_mutable> <batch_size>
bash run.sh force_float32 true 1
echo "GENERATE MODEL SUCCESS!"

###3.infer_python and compute ap
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh <precision> <shape_mutable>
bash run.sh force_float32 true
echo "INFER PYTHON SUCCESS!"