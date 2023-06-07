#!/bin/bash
set -e
set -x

COMPUTE_TOP1_AND_TOP5(){
    PRECISION=$1
    SHAPE_MUTABLE=$2
    BATCH_SIZE=$3
    LANGUAGES=$4
    OUTPUT_DIR=$PROJ_ROOT_PATH/data/output/${LANGUAGES}_output_${PRECISION}_${SHAPE_MUTABLE}_${BATCH_SIZE}
    python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $OUTPUT_DIR/eval_labels.txt \
                                                --result_1_file $OUTPUT_DIR/eval_result_1.txt \
                                                --result_5_file $OUTPUT_DIR/eval_result_5.txt \
                                                --top1andtop5_file $OUTPUT_DIR/eval_result.txt
}
  
cd $PROJ_ROOT_PATH/export_model
bash run.sh
for precision in force_float32 force_float16 qint8_mixed_float16
do
    for shape_mutable in true
    do
        for batch in 1
        do
            cd $PROJ_ROOT_PATH/gen_model
            bash run.sh $precision $shape_mutable $batch 
            cd $PROJ_ROOT_PATH/infer_cpp
            # image_num = 999
            bash run.sh $precision $shape_mutable $batch 50000
            # language = infer_cpp
            COMPUTE_TOP1_AND_TOP5 $precision $shape_mutable $batch infer_cpp
        done
    done
done
