#!/bin/bash
set -e
set -x

languages=infer_python
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision 1
    cd $PROJ_ROOT_PATH/$languages
    bash run.sh $precision 2>&1 | tee $PROJ_ROOT_PATH/data/output/${precision}_log_eval
done
