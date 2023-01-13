#!/bin/bash
set -e
set -x

for parameter_id in 0 #1 2 3 4
do
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh $parameter_id
    for precision in force_float32 force_float16 qint8_mixed_float16
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $parameter_id $precision true 4
        cd $PROJ_ROOT_PATH/infer_python
        bash run.sh $parameter_id $precision true
    done
done
