#!/bin/bash
set -e
set -x

source env.sh
### 0.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh

### 1.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#bash run.sh precision shape_mutable batch_size
bash run.sh force_float32 true 1

### 2.infer
cd $PROJ_ROOT_PATH/infer_python
#bash run.sh precision shape_mutable batch_size image_num
bash run.sh force_float32 true 1 1000

### 3.eval
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_labels.txt \
                                                --result_1_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result_1.txt \
                                                --result_5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result_5.txt \
                                                --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_1/eval_result.txt
