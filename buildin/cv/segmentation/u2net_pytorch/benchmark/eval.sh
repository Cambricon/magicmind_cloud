#!/bin/bash
set -e
set -x

QUANT_MODE=$1
BATCH=$2

#dynamic
if [ $# != 0 ];
then
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $QUANT_MODE $BATCH
else
    echo "Parm doesn't exist, run benchmark"
    cd $PROJ_ROOT_PATH/export_model
    bash run.sh
    for quant_mode in force_float32 force_float16 qint16_mixed_float32
    do
        cd $PROJ_ROOT_PATH/gen_model
        bash run.sh $quant_mode 1
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/infer_python
            bash run.sh $quant_mode $batch
            python $MAGICMIND_CLOUD/test/compare_eval.py --metric u2net --output_file $PROJ_ROOT_PATH/data/output/infer_python_output_${quant_mode}_${batch}/result.txt --output_ok_file $PROJ_ROOT_PATH/data/output_ok/${quant_mode}_${batch}_eval --model u2net_pytorch 
        done
    done
fi
