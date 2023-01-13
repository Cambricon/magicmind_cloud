#!/bin/bash
set -e
set -x

precision=force_float32
shape_mutable=true
image_num=1000

### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh $precision $shape_mutable 1

### 2.infer
cd $PROJ_ROOT_PATH/infer_python
bash run.sh $precision $shape_mutable $image_num

### 3.eval 
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
COMPUTE_TOP1_AND_TOP5 $precision $shape_mutable 1 infer_python
