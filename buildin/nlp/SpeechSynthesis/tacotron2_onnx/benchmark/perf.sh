#!/bin/bash
set -e
set -x

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16
do 
    for batch in 1 32 64
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $precision $batch $batch $batch
        for input_len in 128 256
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $precision $batch $batch $batch $input_len
        done
    done
done