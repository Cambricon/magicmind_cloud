#!/bin/bash
set -e
set -x
echo "Start !"

#1.download datasets and models
cd $PROJ_ROOT_PATH/export_model/
bash run.sh
echo "DOWNLOAD_DATA_SUCCESS!"

#2.build magicmind model
cd $PROJ_ROOT_PATH/gen_model
#Parms 1:precision 2:shape_mutable 3:batch size
bash run.sh force_float32 true 8
echo "GENERATE MODEL SUCCESS!"

#3.infer_cpp
cd $PROJ_ROOT_PATH/infer_cpp
#Parms 1:precision 2:shape_mutable 3:batch size
bash run.sh force_float32 true 8
echo "INFER CPP SUCCESS!"

##4. eval
python $UTILS_PATH/compute_top1_and_top5.py --result_label_file $PROJ_ROOT_PATH/data/output/force_float32_true_8/eval_labels.txt \
                                            --result_1_file $PROJ_ROOT_PATH/data/output/force_float32_true_8/eval_result_1.txt \
                                            --result_5_file $PROJ_ROOT_PATH/data/output/force_float32_true_8/eval_result_5.txt \
                                            --top1andtop5_file $PROJ_ROOT_PATH/data/output/force_float32_true_8/eval_result.txt


echo "COMPARE SUCCESS!"
echo "All has benn Finish!"

