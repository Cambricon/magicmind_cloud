#!/bin/bash
set -e
set -x
echo "Start !"

source env.sh
#1.download datasets and models
cd $PROJ_ROOT_PATH/export_model
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

#2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
bash run.sh force_float32 true 1
echo "GENERATE MODEL SUCCESS!"

#3.infer_python
cd $PROJ_ROOT_PATH/infer_python
bash run.sh force_float32 true 32 
echo "INFER PYTHON SUCCESS!"


### 3.eval
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_32/eval_labels.txt \
                                                --result_1_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_32/eval_result_1.txt \
                                                --result_5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_32/eval_result_5.txt \
                                                --top1andtop5_file $PROJ_ROOT_PATH/data/output/infer_python_output_force_float32_true_32/eval_result.txt
echo "EVAL SUCCESS!"
