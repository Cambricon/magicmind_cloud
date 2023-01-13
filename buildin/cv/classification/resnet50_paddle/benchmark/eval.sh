#!/bin/bash
set -e
set -x

COMPUTE_TOP1_AND_TOP5(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH=$3
    python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH}/eval_labels.txt \
                                                --result_1_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH}/eval_result_1.txt \
                                                --result_5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH}/eval_result_5.txt \
                                                --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_python_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH}/eval_result.txt
}

cd $PROJ_ROOT_PATH/export_model
bash run.sh
#dynamic
for precision in force_float32 force_float16 qint8_mixed_float16
do
    cd $PROJ_ROOT_PATH/gen_model
    bash run.sh $precision true 1	
    for batch in 1
    do
        cd $PROJ_ROOT_PATH/infer_python
        bash run.sh $precision true $batch 1000
        COMPUTE_TOP1_AND_TOP5 $precision true $batch
    done
done
