#!/bin/bash
set -e
set -x

#dynamic
for max_seq_length in 384
do 
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh 1 $max_seq_length
    for quant_mode in force_float32 force_float16
    do 
        cd $PROJ_ROOT_PATH/gen_model
	bash run.sh $quant_mode true 1 $max_seq_length
	for batch in 32
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode true $batch $max_seq_length 
        done
    done
done
