#!/bin/bash
set -e
set -x

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint16_mixed_float32
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 1
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/infer_python
        bash run.sh $precision $batch
    done
done
