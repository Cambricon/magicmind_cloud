#!/bin/bash
set -e
set -x

languages=infer_python
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16
do
    for batch_size in 1
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision $batch_size
        cd $PROJ_ROOT_PATH/$languages
        bash run.sh $precision 2>&1 | tee $PROJ_ROOT_PATH/data/output/${precision}_log_eval
    done
done
