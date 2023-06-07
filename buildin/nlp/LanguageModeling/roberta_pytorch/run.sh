#!/bin/bash
set -e
set -x

source env.sh
#1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh 1 128 ${MODEL_PATH}/roberta_1bs_128.onnx
echo "DOWNLOAD_DATA_SUCCESS!"

#2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh ${MODEL_PATH}/roberta_force_float32_false_1_128 force_float32 1 false 128 ${MODEL_PATH}/roberta_1bs_128.onnx
echo "GENERATE MODEL SUCCESS!"

#3.infer_python
cd $PROJ_ROOT_PATH/infer_python
bash run.sh ${MODEL_PATH}/roberta_force_float32_false_1_128 1 128 ${PROJ_ROOT_PATH}/data/output/fp32_1bs_128
echo "INFER PYTHON SUCCESS!"
