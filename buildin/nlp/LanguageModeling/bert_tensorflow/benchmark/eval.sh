#!/bin/bash
set -e
set -x

for max_seq_length in 384
do 
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for precision in force_float32 force_float16
    do 
        cd $PROJ_ROOT_PATH/gen_model
	bash run.sh $precision true 1 $max_seq_length
	for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $precision true $batch $max_seq_length 
        done
    done
done
