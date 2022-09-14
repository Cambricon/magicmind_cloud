#!/bin/bash
set -e
set -x

cd $PROJ_ROOT_PATH/export_model
bash run.sh
for quant_mode in force_float32 force_float16
do 
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $quant_mode 1 4 8
    for batch in 1 4 8
    do
        for input_len in 128
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode $batch $input_len
        done
    done
done
