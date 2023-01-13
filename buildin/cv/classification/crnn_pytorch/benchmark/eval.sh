#!/bin/bash
set -e
set -x

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision
    for batch_size in 1
    do
        cd $PROJ_ROOT_PATH/infer_python
        bash run.sh $precision $batch_size
    done
done
