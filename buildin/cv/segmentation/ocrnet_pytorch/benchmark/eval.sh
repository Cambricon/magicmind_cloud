#!/bin/bash
set -e
set -x

# export onnx model
cd $PROJ_ROOT_PATH/export_model
bash run.sh

for precision in force_float32 force_float16 qint8_mixed_float16
do
    # generate magicmind model
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 
    # compute mIoU
    cd $PROJ_ROOT_PATH/infer_python
    bash run.sh $precision 2>&1 |tee $PROJ_ROOT_PATH/data/output/ocrnet_${PRECISION}_log_eval
done